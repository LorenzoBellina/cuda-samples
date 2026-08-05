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

#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#define DEBUG_GRID 0
#define DO_TIMING  0

#include <cuda_runtime.h>
#include <helper_functions.h>

#include <cmath>
#include <string>
#include <vector>

#include "particles_kernel.cuh"
#include "vector_functions.h"

// Particle system class
class ParticleSystem
{
public:
    // Allocates host/device buffers and sets default simulation parameters for numParticles particles
    // laid out on a gridSize grid; bUseOpenGL selects VBO-backed storage vs. plain CUDA buffers.
    ParticleSystem(uint numParticles, uint3 gridSize, bool bUseOpenGL);
    // Releases all host and device resources allocated by the constructor.
    ~ParticleSystem();

    enum ParticleConfig { CONFIG_RANDOM, CONFIG_GRID, _NUM_CONFIGS };

    enum ParticleArray {
        POSITION,
        VELOCITY,
    };

    // Advances the simulation by deltaTime: integrates positions, rebuilds the spatial grid,
    // and resolves particle-particle and particle-collider collisions.
    void update(float deltaTime);
    // Reinitializes particle positions/velocities according to the given layout (random cloud or grid).
    void reset(ParticleConfig config);

    // Copies the requested array (positions or velocities) from device to host and returns the host pointer.
    float *getArray(ParticleArray array);
    // Uploads count elements of data into the requested device array starting at index start.
    void   setArray(ParticleArray array, const float *data, int start, int count);

    // Returns the number of particles in the simulation.
    int getNumParticles() const { return m_numParticles; }

    // Returns the OpenGL VBO handle currently holding particle positions.
    unsigned int getCurrentReadBuffer() const { return m_posVbo; }
    // Returns the OpenGL VBO handle holding per-particle colors.
    unsigned int getColorBuffer() const { return m_colorVBO; }

    // Returns the CUDA device pointer backing the position VBO (non-OpenGL mode).
    void *getCudaPosVBO() const { return (void *)m_cudaPosVBO; }
    // Returns the CUDA device pointer backing the color VBO (non-OpenGL mode).
    void *getCudaColorVBO() const { return (void *)m_cudaColorVBO; }

    // Prints the maximum number of particles found in any single grid cell (debugging aid).
    void dumpGrid();
    // Prints position and velocity of count particles starting at start (debugging aid).
    void dumpParticles(uint start, uint count);

    // Sets the number of solver iterations performed per update.
    void setIterations(int i) { m_solverIterations = i; }

    // Sets the global velocity damping factor.
    void setDamping(float x) { m_params.globalDamping = x; }
    // Sets the downward gravity acceleration.
    void setGravity(float x) { m_params.gravity = make_float3(0.0f, x, 0.0f); }

    // Sets the spring stiffness used in particle-particle collision response.
    void setCollideSpring(float x) { m_params.spring = x; }
    // Sets the damping coefficient used in particle-particle collision response.
    void setCollideDamping(float x) { m_params.damping = x; }
    // Sets the tangential shear coefficient used in particle-particle collision response.
    void setCollideShear(float x) { m_params.shear = x; }
    // Sets the attraction coefficient used in particle-particle collision response.
    void setCollideAttraction(float x) { m_params.attraction = x; }

    // Sets the world-space position of the interactive collider sphere.
    void setColliderPos(float3 x) { m_params.colliderPos = x; }

    // Returns the radius of a single particle.
    float  getParticleRadius() { return m_params.particleRadius; }
    // Returns the world-space position of the interactive collider sphere.
    float3 getColliderPos() { return m_params.colliderPos; }
    // Returns the radius of the interactive collider sphere.
    float  getColliderRadius() { return m_params.colliderRadius; }
    // Returns the dimensions (cell counts) of the spatial hashing grid.
    uint3  getGridSize() { return m_params.gridSize; }
    // Returns the world-space origin corresponding to grid cell (0,0,0).
    float3 getWorldOrigin() { return m_params.worldOrigin; }
    // Returns the world-space size of a single grid cell.
    float3 getCellSize() { return m_params.cellSize; }

    // Fills a spherical cluster of particles (radius r cells, given spacing) starting at index,
    // using pos/vel as the sphere's center position and initial velocity.
    void addSphere(int index, float *pos, float *vel, int r, float spacing);

    // Writes the per-phase GPU timings collected by update() (after warm-up) to filename as CSV,
    // one row per (recorded frame, phase) pair: config,density,seed,num_particles,frame,phase,
    // time_ms,num_collisions. "phase" is one of integrate/hash/sort/reorder/collide; num_collisions
    // is the same per-frame value repeated across that frame's 5 phase rows. density is
    // informational only (not applied to the simulation); pass NAN to leave it blank. Appends to
    // filename if it already exists (so multiple runs/configs accumulate in one CSV for later
    // comparison) and only writes the header row for a brand-new file; creates any missing parent
    // directories.
    void writeTimingCSV(const std::string &filename, const std::string &configLabel = "", float density = NAN) const;

    // Sets the RNG seed used by initGrid() so the initial particle layout is reproducible.
    void setSeed(int seed) { m_seed = seed; }

protected: // methods
    ParticleSystem() {}
    // Creates and returns an OpenGL vertex buffer object of the given byte size.
    uint createVBO(uint size);

    // Allocates all host and device buffers sized for numParticles and uploads initial simulation parameters.
    void _initialize(int numParticles);
    // Frees all host and device buffers allocated by _initialize.
    void _finalize();

    // Fills host position/velocity arrays with a regular lattice of numParticles particles,
    // spaced by spacing and randomly perturbed by up to jitter.
    void initGrid(uint *size, float spacing, float jitter, uint numParticles);

protected: // data
    bool m_bInitialized, m_bUseOpenGL;
    uint m_numParticles;

    // CPU data
    float *m_hPos; // particle positions
    float *m_hVel; // particle velocities

    uint *m_hParticleHash;
    uint *m_hCellStart;
    uint *m_hCellEnd;

    // GPU data
    float *m_dPos;
    float *m_dVel;

    float *m_dSortedPos;
    float *m_dSortedVel;

    // grid data for sorting method
    uint *m_dGridParticleHash;  // grid hash value for each particle
    uint *m_dGridParticleIndex; // particle index for each particle
    uint *m_dCellStart;         // index of start of each cell in sorted list
    uint *m_dCellEnd;           // index of end of cell

    uint m_gridSortBits;

    uint m_posVbo;   // vertex buffer object for particle positions
    uint m_colorVBO; // vertex buffer object for colors

    float *m_cudaPosVBO;   // these are the CUDA deviceMem Pos
    float *m_cudaColorVBO; // these are the CUDA deviceMem Color

    struct cudaGraphicsResource *m_cuda_posvbo_resource;   // handles OpenGL-CUDA exchange
    struct cudaGraphicsResource *m_cuda_colorvbo_resource; // handles OpenGL-CUDA exchange

    // params
    SimParams m_params;
    uint3     m_gridSize;
    uint      m_numGridCells;

    StopWatchInterface *m_timer;

    uint m_solverIterations;

    // RNG seed used by initGrid(); defaults to the sample's original hardcoded value.
    int m_seed = 1973;

    // per-phase GPU timing (update() breakdown)
    enum TimingPhase { TIMING_INTEGRATE = 0, TIMING_HASH, TIMING_SORT, TIMING_REORDER, TIMING_COLLIDE, TIMING_NUM_PHASES };

    // First N frames are discarded from the timing measurements (CUDA context/JIT warm-up),
    // but still simulated normally.
    static constexpr unsigned int kTimingWarmupFrames = 20;

    cudaEvent_t         m_timingStart[TIMING_NUM_PHASES];
    cudaEvent_t         m_timingStop[TIMING_NUM_PHASES];
    std::vector<float>  m_timingMs[TIMING_NUM_PHASES];
    unsigned int         m_frameIndex;

    // Per-frame particle-particle collision counts (see COUNT_COLLISIONS in
    // particles_kernel_impl.cuh), recorded 1:1 alongside m_timingMs (same warm-up skip, same
    // frame indexing).
    std::vector<unsigned int> m_collisionCounts;
};

#endif // __PARTICLESYSTEM_H__
