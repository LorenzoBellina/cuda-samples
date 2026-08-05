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

// This file contains C wrappers around the some of the CUDA API and the
// kernel functions so that they can be called from "particleSystem.cpp"

#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <string.h>

#include "particles_kernel_impl.cuh"
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

extern "C"
{

    // Picks a CUDA device (respecting a -device= argument, otherwise the fastest available)
    // and exits the process if no CUDA-capable device is found.
    void cudaInit(int argc, char **argv)
    {
        int devID;

        // use command-line specified CUDA device, otherwise use device with highest
        // Gflops/s
        devID = findCudaDevice(argc, (const char **)argv);

        if (devID < 0) {
            printf("No CUDA Capable devices found, exiting...\n");
            exit(EXIT_SUCCESS);
        }
    }

    // Allocates a device buffer of size bytes via cudaMalloc.
    void allocateArray(void **devPtr, size_t size) { checkCudaErrors(cudaMalloc(devPtr, size)); }

    // Frees a device buffer previously returned by allocateArray.
    void freeArray(void *devPtr) { checkCudaErrors(cudaFree(devPtr)); }

    // Blocks the host until all previously issued device work has completed.
    void threadSync() { checkCudaErrors(cudaDeviceSynchronize()); }

    // Copies size bytes from host memory into device memory at byte offset offset.
    void copyArrayToDevice(void *device, const void *host, int offset, int size)
    {
        checkCudaErrors(cudaMemcpy((char *)device + offset, host, size, cudaMemcpyHostToDevice));
    }

    // Registers an OpenGL buffer object with the CUDA/GL interop so it can later be mapped
    // as a CUDA device pointer.
    void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone));
    }

    // Unregisters a CUDA/GL interop resource previously created by registerGLBufferObject.
    void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));
    }

    // Maps a registered OpenGL buffer for CUDA access and returns the resulting device pointer.
    void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource)
    {
        void *ptr;
        checkCudaErrors(cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
        size_t num_bytes;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes, *cuda_vbo_resource));
        return ptr;
    }

    // Unmaps a CUDA/GL interop resource previously mapped by mapGLBufferObject, handing it back to OpenGL.
    void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
    {
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
    }

    // Copies size bytes from device memory (mapping cuda_vbo_resource first if it is non-null)
    // into host memory, unmapping the resource again afterward.
    void copyArrayFromDevice(void *host, const void *device, struct cudaGraphicsResource **cuda_vbo_resource, int size)
    {
        if (cuda_vbo_resource) {
            device = mapGLBufferObject(cuda_vbo_resource);
        }

        checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));

        if (cuda_vbo_resource) {
            unmapGLBufferObject(*cuda_vbo_resource);
        }
    }

    // Uploads the simulation parameters struct into the cudaParams __constant__ symbol so
    // every kernel can read it without an extra pointer argument.
    void setParameters(SimParams *hostParams)
    {
        // copy parameters to constant memory
        checkCudaErrors(cudaMemcpyToSymbol(cudaParams, hostParams, sizeof(SimParams)));
    }

    // Round a / b to nearest higher integer value
    uint iDivUp(uint a, uint b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

    // compute grid and thread block size for a given number of elements
    void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
    {
        numThreads = min(blockSize, n);
        numBlocks  = iDivUp(n, numThreads);
    }

    // Advances every particle's position/velocity by deltaTime using the integrate_functor
    // (applies gravity and damping, then simple boundary-wall collisions), executed via a
    // Thrust for_each over zipped position/velocity device iterators.
    void integrateSystem(float *pos, float *vel, float deltaTime, uint numParticles)
    {
        thrust::device_ptr<float4> d_pos4((float4 *)pos);
        thrust::device_ptr<float4> d_vel4((float4 *)vel);

        thrust::for_each(thrust::make_zip_iterator(d_pos4, d_vel4),
                         thrust::make_zip_iterator(d_pos4 + numParticles, d_vel4 + numParticles),
                         integrate_functor(deltaTime));
    }

    // Launches calcHashD to compute, for every particle, the linear hash of the grid cell it
    // currently occupies (from its position) plus its own particle index; these pairs are the
    // input to the subsequent sort-by-cell step.
    void calcHash(uint *gridParticleHash, uint *gridParticleIndex, float *pos, int numParticles)
    {
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // execute the kernel
        calcHashD<<<numBlocks, numThreads>>>(gridParticleHash, gridParticleIndex, (float4 *)pos, numParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }

    void reorderDataAndFindCellStart(uint  *cellStart,
                                     uint  *cellEnd,
                                     float *sortedPos,
                                     float *sortedVel,
                                     uint  *gridParticleHash,
                                     uint  *gridParticleIndex,
                                     float *oldPos,
                                     float *oldVel,
                                     uint   numParticles,
                                     uint   numCells)
    {
        // Resets every cell to "empty" (0xffffffff), then launches reorderDataAndFindCellStartD,
        // which uses shared memory to detect, per sorted particle, where a new cell hash begins;
        // it records each cell's start/end index and writes positions/velocities into sorted order.
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // set all cells to empty
        checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells * sizeof(uint)));

        uint smemSize = sizeof(uint) * (numThreads + 1);
        reorderDataAndFindCellStartD<<<numBlocks, numThreads, smemSize>>>(cellStart,
                                                                          cellEnd,
                                                                          (float4 *)sortedPos,
                                                                          (float4 *)sortedVel,
                                                                          gridParticleHash,
                                                                          gridParticleIndex,
                                                                          (float4 *)oldPos,
                                                                          (float4 *)oldVel,
                                                                          numParticles);
        getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");
    }

    void collide(float *newVel,
                 float *sortedPos,
                 float *sortedVel,
                 uint  *gridParticleIndex,
                 uint  *cellStart,
                 uint  *cellEnd,
                 uint   numParticles,
                 uint   numCells,
                 unsigned int *numCollisions)
    {
        // Launches collideD, one thread per particle: each thread scans its 3x3x3 block of
        // neighboring grid cells (via cellStart/cellEnd) reading sortedPos/sortedVel directly
        // from global memory, resolves DEM-style collisions with nearby particles and the
        // interactive collider sphere, and writes the resulting velocity to newVel at the
        // particle's original (pre-sort) index.
        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 64, numBlocks, numThreads);

#if COUNT_COLLISIONS
        // Reset the instrumentation-only collision counter (see particles_kernel_impl.cuh)
        // before each launch so it reflects this frame only.
        unsigned int zero = 0;
        checkCudaErrors(cudaMemcpyToSymbol(d_collisionCount, &zero, sizeof(unsigned int)));
#endif

        // execute the kernel
        collideD<<<numBlocks, numThreads>>>((float4 *)newVel,
                                            (float4 *)sortedPos,
                                            (float4 *)sortedVel,
                                            gridParticleIndex,
                                            cellStart,
                                            cellEnd,
                                            numParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");

#if COUNT_COLLISIONS
        checkCudaErrors(cudaMemcpyFromSymbol(numCollisions, d_collisionCount, sizeof(unsigned int)));
#else
        *numCollisions = 0;
#endif
    }

    // Sorts particle indices by their grid cell hash using Thrust's sort_by_key, so that after
    // sorting, particles occupying the same cell become contiguous in dGridParticleIndex.
    void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles)
    {
        thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
                            thrust::device_ptr<uint>(dGridParticleHash + numParticles),
                            thrust::device_ptr<uint>(dGridParticleIndex));
    }

} // extern "C"
