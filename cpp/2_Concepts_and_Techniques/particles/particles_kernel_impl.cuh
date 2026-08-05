/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * CUDA particle system kernel code.
 */

#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include <cooperative_groups.h>
#include <math.h>
#include <stdio.h>

#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

// for cuda::std::get
#include <cuda/std/utility>

namespace cg = cooperative_groups;
#include "helper_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"

// simulation parameters in constant memory
__constant__ SimParams cudaParams;

// Instrumentation-only collision counter feeding the num_collisions column of the benchmark
// CSV (see ParticleSystem::writeTimingCSV / collide() in particleSystem_cuda.cu). A single
// atomicAdd per detected particle-particle overlap is negligible next to the rest of the
// collision kernel, so it stays ON by default (including in --benchmark mode). To exclude it
// from "clean" (non-instrumented) builds, define COUNT_COLLISIONS=0 for the compiler
// (e.g. nvcc -DCOUNT_COLLISIONS=0), which compiles the counter and its atomics out entirely.
#ifndef COUNT_COLLISIONS
#define COUNT_COLLISIONS 1
#endif

#if COUNT_COLLISIONS
__device__ unsigned int d_collisionCount;
#endif

// Thrust functor (applied per-particle via for_each over zipped position/velocity iterators)
// that performs the Euler integration step of the simulation.
struct integrate_functor
{
    float deltaTime;

    // Captures the timestep to integrate over.
    __host__ __device__ integrate_functor(float delta_time)
        : deltaTime(delta_time)
    {
    }

    // Given a (position, velocity) tuple t, applies gravity and global damping to the
    // velocity, advances the position by velocity * deltaTime, clamps the particle to stay
    // within the [-1,1] cube (damping the velocity component on bounce), and writes the
    // updated position/velocity back into t.
    template <typename Tuple> __device__ void operator()(Tuple t)
    {
        volatile float4 posData = cuda::std::get<0>(t);
        volatile float4 velData = cuda::std::get<1>(t);
        float3          pos     = make_float3(posData.x, posData.y, posData.z);
        float3          vel     = make_float3(velData.x, velData.y, velData.z);

        vel += cudaParams.gravity * deltaTime;
        vel *= cudaParams.globalDamping;

        // new position = old position + velocity * deltaTime
        pos += vel * deltaTime;

// set this to zero to disable collisions with cube sides
#if 1

        if (pos.x > 1.0f - cudaParams.particleRadius) {
            pos.x = 1.0f - cudaParams.particleRadius;
            vel.x *= cudaParams.boundaryDamping;
        }

        if (pos.x < -1.0f + cudaParams.particleRadius) {
            pos.x = -1.0f + cudaParams.particleRadius;
            vel.x *= cudaParams.boundaryDamping;
        }

        if (pos.y > 1.0f - cudaParams.particleRadius) {
            pos.y = 1.0f - cudaParams.particleRadius;
            vel.y *= cudaParams.boundaryDamping;
        }

        if (pos.z > 1.0f - cudaParams.particleRadius) {
            pos.z = 1.0f - cudaParams.particleRadius;
            vel.z *= cudaParams.boundaryDamping;
        }

        if (pos.z < -1.0f + cudaParams.particleRadius) {
            pos.z = -1.0f + cudaParams.particleRadius;
            vel.z *= cudaParams.boundaryDamping;
        }

#endif

        if (pos.y < -1.0f + cudaParams.particleRadius) {
            pos.y = -1.0f + cudaParams.particleRadius;
            vel.y *= cudaParams.boundaryDamping;
        }

        // store new position and velocity
        cuda::std::get<0>(t) = make_float4(pos, posData.w);
        cuda::std::get<1>(t) = make_float4(vel, velData.w);
    }
};

// Converts a world-space position p into (x,y,z) integer coordinates of the uniform grid
// cell that contains it, relative to worldOrigin and scaled by cellSize.
__device__ int3 calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = floorf((p.x - cudaParams.worldOrigin.x) / cudaParams.cellSize.x);
    gridPos.y = floorf((p.y - cudaParams.worldOrigin.y) / cudaParams.cellSize.y);
    gridPos.z = floorf((p.z - cudaParams.worldOrigin.z) / cudaParams.cellSize.z);
    return gridPos;
}

// Wraps gridPos's coordinates into the grid bounds (grid size is assumed to be a power of two,
// so wrapping is a bitmask) and flattens them into a single linear cell hash/index.
__device__ uint calcGridHash(int3 gridPos)
{
    gridPos.x = gridPos.x & (cudaParams.gridSize.x - 1); // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y & (cudaParams.gridSize.y - 1);
    gridPos.z = gridPos.z & (cudaParams.gridSize.z - 1);
    return __umul24(__umul24(gridPos.z, cudaParams.gridSize.y), cudaParams.gridSize.x)
         + __umul24(gridPos.y, cudaParams.gridSize.x) + gridPos.x;
}

// Kernel: for each particle (one thread per particle), computes the grid cell it currently
// occupies and writes out that cell's hash plus the particle's own index; these paired
// arrays are subsequently sorted by hash to group particles by cell.
__global__ void calcHashD(uint   *gridParticleHash,  // output
                          uint   *gridParticleIndex, // output
                          float4 *pos,               // input: positions
                          uint    numParticles)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles)
        return;

    volatile float4 p = pos[index];

    // get address in grid
    int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
    uint hash    = calcGridHash(gridPos);

    // store grid hash and particle index
    gridParticleHash[index]  = hash;
    gridParticleIndex[index] = index;
}

// Kernel: given particles already sorted by grid-cell hash (gridParticleHash/gridParticleIndex),
// reorders their position/velocity data into that sorted order (sortedPos/sortedVel) and records,
// for each grid cell, the index range [cellStart, cellEnd) of particles belonging to it. Uses
// shared memory to let each thread compare its hash against its immediate neighbor's hash
// (loaded once per block) rather than every thread re-reading two global-memory hash values.
__global__ void reorderDataAndFindCellStartD(uint   *cellStart,         // output: cell start index
                                             uint   *cellEnd,           // output: cell end index
                                             float4 *sortedPos,         // output: sorted positions
                                             float4 *sortedVel,         // output: sorted velocities
                                             uint   *gridParticleHash,  // input: sorted grid hashes
                                             uint   *gridParticleIndex, // input: sorted particle indices
                                             float4 *oldPos,            // input: sorted position array
                                             float4 *oldVel,            // input: sorted velocity array
                                             uint    numParticles)
{
    // Handle to thread block group
    cg::thread_block       cta = cg::this_thread_block();
    extern __shared__ uint sharedHash[]; // blockSize + 1 elements
    uint                   index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    uint hash;

    // handle case when no. of particles not multiple of block size
    if (index < numParticles) {
        hash = gridParticleHash[index];

        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x + 1] = hash;

        if (index > 0 && threadIdx.x == 0) {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = gridParticleHash[index - 1];
        }
    }

    cg::sync(cta);

    if (index < numParticles) {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell

        if (index == 0 || hash != sharedHash[threadIdx.x]) {
            cellStart[hash] = index;

            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
        }

        if (index == numParticles - 1) {
            cellEnd[hash] = index + 1;
        }

        // Now use the sorted index to reorder the pos and vel data
        uint   sortedIndex = gridParticleIndex[index];
        float4 pos         = oldPos[sortedIndex];
        float4 vel         = oldVel[sortedIndex];

        sortedPos[index] = pos;
        sortedVel[index] = vel;
    }
}

// Computes the Discrete Element Method (DEM) contact force between two spheres A and B
// (given their positions, velocities, and radii): zero if they don't overlap, otherwise a
// spring force pushing them apart, a damping force opposing relative motion, a tangential
// shear force, and an attraction term pulling them together. Returns the force to apply to A.
__device__ float3
collideSpheres(float3 posA, float3 posB, float3 velA, float3 velB, float radiusA, float radiusB, float attraction)
{
    // calculate relative position
    float3 relPos = posB - posA;

    float dist        = length(relPos);
    float collideDist = radiusA + radiusB;

    float3 force = make_float3(0.0f);

    if (dist < collideDist) {
        float3 norm = relPos / dist;

        // relative velocity
        float3 relVel = velB - velA;

        // relative tangential velocity
        float3 tanVel = relVel - (dot(relVel, norm) * norm);

        // spring force
        force = -cudaParams.spring * (collideDist - dist) * norm;
        // dashpot (damping) force
        force += cudaParams.damping * relVel;
        // tangential shear force
        force += cudaParams.shear * tanVel;
        // attraction
        force += attraction * relPos;
    }

    return force;
}

// Accumulates the total DEM collision force on particle `index` (at position pos, velocity
// vel) from every other particle located in the single grid cell gridPos. Looks up the
// cell's particle range via cellStart/cellEnd (both read from global memory) and reads each
// candidate neighbor's position/velocity directly out of the global oldPos/oldVel arrays
// (sorted by cell) rather than staging them in shared memory.
__device__ float3 collideCell(int3    gridPos,
                              uint    index,
                              float3  pos,
                              float3  vel,
                              float4 *oldPos,
                              float4 *oldVel,
                              uint   *cellStart,
                              uint   *cellEnd)
{
    uint gridHash = calcGridHash(gridPos);

    // get start of bucket for this cell
    uint startIndex = cellStart[gridHash];

    float3 force = make_float3(0.0f);

    if (startIndex != 0xffffffff) // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = cellEnd[gridHash];

        for (uint j = startIndex; j < endIndex; j++) {
            if (j != index) // check not colliding with self
            {
                float3 pos2 = make_float3(oldPos[j]);
                float3 vel2 = make_float3(oldVel[j]);

#if COUNT_COLLISIONS
                // Instrumentation only: re-checks the same overlap predicate collideSpheres()
                // uses internally, just to count events for the CSV. Each thread scans its
                // neighbors independently, so an overlapping pair is counted twice (once from
                // each particle's side) -- consistent with how the force computation below
                // also revisits every pair from both sides.
                if (length(pos2 - pos) < (cudaParams.particleRadius + cudaParams.particleRadius)) {
                    atomicAdd(&d_collisionCount, 1u);
                }
#endif

                // collide two spheres
                force += collideSpheres(
                    pos, pos2, vel, vel2, cudaParams.particleRadius, cudaParams.particleRadius, cudaParams.attraction);
            }
        }
    }

    return force;
}

// Kernel: one thread per particle. Reads the particle's sorted position/velocity, determines
// its grid cell, then examines all 27 neighboring cells (3x3x3 block, itself included) via
// collideCell to accumulate particle-particle collision forces, plus one extra collision
// check against the interactive collider sphere. The resulting velocity is written back to
// newVel at the particle's original (pre-sort) index via gridParticleIndex.
__global__ void collideD(float4 *newVel,            // output: new velocity
                         float4 *oldPos,            // input: sorted positions
                         float4 *oldVel,            // input: sorted velocities
                         uint   *gridParticleIndex, // input: sorted particle indices
                         uint   *cellStart,
                         uint   *cellEnd,
                         uint    numParticles)
{
    uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles)
        return;

    // read particle data from sorted arrays
    float3 pos = make_float3(oldPos[index]);
    float3 vel = make_float3(oldVel[index]);

    // get address in grid
    int3 gridPos = calcGridPos(pos);

    // examine neighbouring cells
    float3 force = make_float3(0.0f);

    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                force += collideCell(neighbourPos, index, pos, vel, oldPos, oldVel, cellStart, cellEnd);
            }
        }
    }

    // collide with cursor sphere
    force += collideSpheres(pos,
                            cudaParams.colliderPos,
                            vel,
                            make_float3(0.0f, 0.0f, 0.0f),
                            cudaParams.particleRadius,
                            cudaParams.colliderRadius,
                            0.0f);

    // write new velocity back to original unsorted location
    uint originalIndex    = gridParticleIndex[index];
    newVel[originalIndex] = make_float4(vel + force, 0.0f);
}

#endif
