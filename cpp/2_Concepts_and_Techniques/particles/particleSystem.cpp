/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// OpenGL Graphics includes
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include "particleSystem.h"

#include <algorithm>
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <filesystem>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_gl.h>
#include <math.h>
#include <memory.h>

#include "particleSystem.cuh"
#include "particles_kernel.cuh"

#ifndef CUDART_PI_F
#define CUDART_PI_F 3.141592654f
#endif

// Constructs the simulation: stores the requested particle count and grid size, sets the
// default physical/collision parameters, then allocates host/device buffers via _initialize().
ParticleSystem::ParticleSystem(uint numParticles, uint3 gridSize, bool bUseOpenGL)
    : m_bInitialized(false)
    , m_bUseOpenGL(bUseOpenGL)
    , m_numParticles(numParticles)
    , m_hPos(0)
    , m_hVel(0)
    , m_dPos(0)
    , m_dVel(0)
    , m_gridSize(gridSize)
    , m_timer(NULL)
    , m_solverIterations(1)
    , m_frameIndex(0)
{
    m_numGridCells = m_gridSize.x * m_gridSize.y * m_gridSize.z;
    //    float3 worldSize = make_float3(2.0f, 2.0f, 2.0f);

    m_gridSortBits = 18; // increase this for larger grids

    // set simulation parameters
    m_params.gridSize  = m_gridSize;
    m_params.numCells  = m_numGridCells;
    m_params.numBodies = m_numParticles;

    m_params.particleRadius = 1.0f / 64.0f;
    m_params.colliderPos    = make_float3(-1.2f, -0.8f, 0.8f);
    m_params.colliderRadius = 0.2f;

    m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
    //    m_params.cellSize = make_float3(worldSize.x / m_gridSize.x, worldSize.y / m_gridSize.y, worldSize.z /
    //    m_gridSize.z);
    float cellSize    = m_params.particleRadius * 2.0f; // cell size equal to particle diameter
    m_params.cellSize = make_float3(cellSize, cellSize, cellSize);

    m_params.spring          = 0.5f;
    m_params.damping         = 0.02f;
    m_params.shear           = 0.1f;
    m_params.attraction      = 0.0f;
    m_params.boundaryDamping = -0.5f;

    m_params.gravity       = make_float3(0.0f, -0.0003f, 0.0f);
    m_params.globalDamping = 1.0f;

    _initialize(numParticles);
}

// Destroys the simulation, releasing every host/device buffer allocated for it.
ParticleSystem::~ParticleSystem()
{
    _finalize();
    m_numParticles = 0;
}

// Allocates an OpenGL array buffer of the given byte size and returns its handle.
uint ParticleSystem::createVBO(uint size)
{
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return vbo;
}

// Linearly interpolates between a and b by fraction t.
inline float lerp(float a, float b, float t) { return a + t * (b - a); }

// Maps t in [0,1] onto a rainbow gradient (red -> orange -> yellow -> green -> cyan -> blue -> magenta),
// writing the interpolated RGB result into r.
void colorRamp(float t, float *r)
{
    const int ncolors       = 7;
    float     c[ncolors][3] = {
        {
            1.0,
            0.0,
            0.0,
        },
        {
            1.0,
            0.5,
            0.0,
        },
        {
            1.0,
            1.0,
            0.0,
        },
        {
            0.0,
            1.0,
            0.0,
        },
        {
            0.0,
            1.0,
            1.0,
        },
        {
            0.0,
            0.0,
            1.0,
        },
        {
            1.0,
            0.0,
            1.0,
        },
    };
    t       = t * (ncolors - 1);
    int   i = (int)t;
    float u = t - floorf(t);
    r[0]    = lerp(c[i][0], c[i + 1][0], u);
    r[1]    = lerp(c[i][1], c[i + 1][1], u);
    r[2]    = lerp(c[i][2], c[i + 1][2], u);
}

// Allocates host arrays (positions, velocities, cell start/end tables) and their device
// counterparts (including sorted-position/velocity and grid-hash scratch buffers), creates
// the position/color VBOs (or plain CUDA buffers when not using OpenGL), and uploads
// the simulation parameters to constant memory.
void ParticleSystem::_initialize(int numParticles)
{
    assert(!m_bInitialized);

    m_numParticles = numParticles;

    // allocate host storage
    m_hPos = new float[m_numParticles * 4];
    m_hVel = new float[m_numParticles * 4];
    memset(m_hPos, 0, m_numParticles * 4 * sizeof(float));
    memset(m_hVel, 0, m_numParticles * 4 * sizeof(float));

    m_hCellStart = new uint[m_numGridCells];
    memset(m_hCellStart, 0, m_numGridCells * sizeof(uint));

    m_hCellEnd = new uint[m_numGridCells];
    memset(m_hCellEnd, 0, m_numGridCells * sizeof(uint));

    // allocate GPU data
    unsigned int memSize = sizeof(float) * 4 * m_numParticles;

    if (m_bUseOpenGL) {
        m_posVbo = createVBO(memSize);
        registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
    }
    else {
        checkCudaErrors(cudaMalloc((void **)&m_cudaPosVBO, memSize));
    }

    allocateArray((void **)&m_dVel, memSize);

    allocateArray((void **)&m_dSortedPos, memSize);
    allocateArray((void **)&m_dSortedVel, memSize);

    allocateArray((void **)&m_dGridParticleHash, m_numParticles * sizeof(uint));
    allocateArray((void **)&m_dGridParticleIndex, m_numParticles * sizeof(uint));

    allocateArray((void **)&m_dCellStart, m_numGridCells * sizeof(uint));
    allocateArray((void **)&m_dCellEnd, m_numGridCells * sizeof(uint));

    if (m_bUseOpenGL) {
        m_colorVBO = createVBO(m_numParticles * 4 * sizeof(float));
        registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);

        // fill color buffer
        glBindBuffer(GL_ARRAY_BUFFER, m_colorVBO);
        float *data = (float *)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        float *ptr  = data;

        for (uint i = 0; i < m_numParticles; i++) {
            float t = i / (float)m_numParticles;
#if 0
            *ptr++ = rand() / (float) RAND_MAX;
            *ptr++ = rand() / (float) RAND_MAX;
            *ptr++ = rand() / (float) RAND_MAX;
#else
            colorRamp(t, ptr);
            ptr += 3;
#endif
            *ptr++ = 1.0f;
        }

        glUnmapBuffer(GL_ARRAY_BUFFER);
    }
    else {
        checkCudaErrors(cudaMalloc((void **)&m_cudaColorVBO, sizeof(float) * numParticles * 4));
    }

    sdkCreateTimer(&m_timer);

    for (int p = 0; p < TIMING_NUM_PHASES; ++p) {
        checkCudaErrors(cudaEventCreate(&m_timingStart[p]));
        checkCudaErrors(cudaEventCreate(&m_timingStop[p]));
    }

    setParameters(&m_params);

    m_bInitialized = true;
}

// Frees every host and device buffer allocated in _initialize(), and deletes/unregisters
// the OpenGL VBOs (or frees the plain CUDA buffers) depending on m_bUseOpenGL.
void ParticleSystem::_finalize()
{
    assert(m_bInitialized);

    delete[] m_hPos;
    delete[] m_hVel;
    delete[] m_hCellStart;
    delete[] m_hCellEnd;

    freeArray(m_dVel);
    freeArray(m_dSortedPos);
    freeArray(m_dSortedVel);

    freeArray(m_dGridParticleHash);
    freeArray(m_dGridParticleIndex);
    freeArray(m_dCellStart);
    freeArray(m_dCellEnd);

    for (int p = 0; p < TIMING_NUM_PHASES; ++p) {
        checkCudaErrors(cudaEventDestroy(m_timingStart[p]));
        checkCudaErrors(cudaEventDestroy(m_timingStop[p]));
    }

    if (m_bUseOpenGL) {
        unregisterGLBufferObject(m_cuda_colorvbo_resource);
        unregisterGLBufferObject(m_cuda_posvbo_resource);
        glDeleteBuffers(1, (const GLuint *)&m_posVbo);
        glDeleteBuffers(1, (const GLuint *)&m_colorVBO);
    }
    else {
        checkCudaErrors(cudaFree(m_cudaPosVBO));
        checkCudaErrors(cudaFree(m_cudaColorVBO));
    }
}

// Advances the simulation by one step of deltaTime: integrates positions/velocities under
// gravity and damping, recomputes each particle's grid cell hash, sorts particles by hash,
// reorders position/velocity data into sorted order and finds each cell's start/end range,
// then resolves particle-particle and particle-collider collisions.
void ParticleSystem::update(float deltaTime)
{
    assert(m_bInitialized);

    float *dPos;

    if (m_bUseOpenGL) {
        dPos = (float *)mapGLBufferObject(&m_cuda_posvbo_resource);
    }
    else {
        dPos = (float *)m_cudaPosVBO;
    }

    // update constants
    setParameters(&m_params);

    const bool bRecordTiming = (m_frameIndex >= kTimingWarmupFrames); //skips the first 20 frames
    float      phaseMs[TIMING_NUM_PHASES];

    // integrate
    checkCudaErrors(cudaEventRecord(m_timingStart[TIMING_INTEGRATE]));
    integrateSystem(dPos, m_dVel, deltaTime, m_numParticles);
    checkCudaErrors(cudaEventRecord(m_timingStop[TIMING_INTEGRATE]));
    checkCudaErrors(cudaEventSynchronize(m_timingStop[TIMING_INTEGRATE]));
    checkCudaErrors(
        cudaEventElapsedTime(&phaseMs[TIMING_INTEGRATE], m_timingStart[TIMING_INTEGRATE], m_timingStop[TIMING_INTEGRATE]));

    // calculate grid hash
    checkCudaErrors(cudaEventRecord(m_timingStart[TIMING_HASH]));
    calcHash(m_dGridParticleHash, m_dGridParticleIndex, dPos, m_numParticles);
    checkCudaErrors(cudaEventRecord(m_timingStop[TIMING_HASH]));
    checkCudaErrors(cudaEventSynchronize(m_timingStop[TIMING_HASH]));
    checkCudaErrors(cudaEventElapsedTime(&phaseMs[TIMING_HASH], m_timingStart[TIMING_HASH], m_timingStop[TIMING_HASH]));

    // sort particles based on hash
    checkCudaErrors(cudaEventRecord(m_timingStart[TIMING_SORT]));
    sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_numParticles);
    checkCudaErrors(cudaEventRecord(m_timingStop[TIMING_SORT]));
    checkCudaErrors(cudaEventSynchronize(m_timingStop[TIMING_SORT]));
    checkCudaErrors(cudaEventElapsedTime(&phaseMs[TIMING_SORT], m_timingStart[TIMING_SORT], m_timingStop[TIMING_SORT]));

    // reorder particle arrays into sorted order and
    // find start and end of each cell
    checkCudaErrors(cudaEventRecord(m_timingStart[TIMING_REORDER]));
    reorderDataAndFindCellStart(m_dCellStart,
                                m_dCellEnd,
                                m_dSortedPos,
                                m_dSortedVel,
                                m_dGridParticleHash,
                                m_dGridParticleIndex,
                                dPos,
                                m_dVel,
                                m_numParticles,
                                m_numGridCells);
    checkCudaErrors(cudaEventRecord(m_timingStop[TIMING_REORDER]));
    checkCudaErrors(cudaEventSynchronize(m_timingStop[TIMING_REORDER]));
    checkCudaErrors(
        cudaEventElapsedTime(&phaseMs[TIMING_REORDER], m_timingStart[TIMING_REORDER], m_timingStop[TIMING_REORDER]));

    // process collisions
    unsigned int frameCollisions = 0;
    checkCudaErrors(cudaEventRecord(m_timingStart[TIMING_COLLIDE]));
    collide(m_dVel,
            m_dSortedPos,
            m_dSortedVel,
            m_dGridParticleIndex,
            m_dCellStart,
            m_dCellEnd,
            m_numParticles,
            m_numGridCells,
            &frameCollisions);
    checkCudaErrors(cudaEventRecord(m_timingStop[TIMING_COLLIDE]));
    checkCudaErrors(cudaEventSynchronize(m_timingStop[TIMING_COLLIDE]));
    checkCudaErrors(
        cudaEventElapsedTime(&phaseMs[TIMING_COLLIDE], m_timingStart[TIMING_COLLIDE], m_timingStop[TIMING_COLLIDE]));

    // discard warm-up iterations from the recorded measurements only; the simulation itself
    // still runs unmodified for every frame above
    if (bRecordTiming) {
        for (int p = 0; p < TIMING_NUM_PHASES; ++p) {
            m_timingMs[p].push_back(phaseMs[p]);
        }

        m_collisionCounts.push_back(frameCollisions);
    }

    ++m_frameIndex;

    // note: do unmap at end here to avoid unnecessary graphics/CUDA context switch
    if (m_bUseOpenGL) {
        unmapGLBufferObject(m_cuda_posvbo_resource);
    }
}

// Writes the accumulated per-phase timings (populated by update(), excluding warm-up frames)
// to filename as CSV, one row per (recorded frame, phase) pair. Creates any missing parent
// directories; appends to filename if it already exists (so multiple runs/configs accumulate
// in one cumulative CSV for later comparison) and only writes the header row for a brand-new
// file. No-op (with a stderr warning) if the file can't be opened, or if no frames were
// recorded yet (e.g. the run didn't exceed the warm-up window).
void ParticleSystem::writeTimingCSV(const std::string &filename, const std::string &configLabel, float density) const
{
    const std::vector<float> &integrateMs = m_timingMs[TIMING_INTEGRATE];

    if (integrateMs.empty()) {
        fprintf(stderr, "writeTimingCSV: no timing samples recorded, skipping %s\n", filename.c_str());
        return;
    }

    static const char *const kPhaseNames[TIMING_NUM_PHASES] = {"integrate", "hash", "sort", "reorder", "collide"};

    std::filesystem::path filePath(filename);

    if (filePath.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(filePath.parent_path(), ec);
    }

    bool fileExists = std::filesystem::exists(filePath);

    FILE *fp = fopen(filename.c_str(), "a");

    if (!fp) {
        fprintf(stderr, "writeTimingCSV: unable to open %s for writing\n", filename.c_str());
        return;
    }

    if (!fileExists) {
        fprintf(fp, "config,density,seed,num_particles,frame,phase,time_ms,num_collisions\n");
    }

    char densityField[32];

    if (std::isnan(density)) {
        densityField[0] = '\0';
    }
    else {
        snprintf(densityField, sizeof(densityField), "%f", density);
    }

    for (size_t i = 0; i < integrateMs.size(); ++i) {
        unsigned int collisions = (i < m_collisionCounts.size()) ? m_collisionCounts[i] : 0;

        for (int p = 0; p < TIMING_NUM_PHASES; ++p) {
            fprintf(fp,
                    "\"%s\",%s,%d,%u,%zu,%s,%f,%u\n",
                    configLabel.c_str(),
                    densityField,
                    m_seed,
                    m_numParticles,
                    i,
                    kPhaseNames[p],
                    m_timingMs[p][i],
                    collisions);
        }
    }

    fclose(fp);
}

// Downloads the cell start/end tables to the host and prints the largest occupancy found
// in any single grid cell; useful for sanity-checking the spatial hash and grid resolution.
void ParticleSystem::dumpGrid()
{
    // dump grid information
    copyArrayFromDevice(m_hCellStart, m_dCellStart, 0, sizeof(uint) * m_numGridCells);
    copyArrayFromDevice(m_hCellEnd, m_dCellEnd, 0, sizeof(uint) * m_numGridCells);
    uint maxCellSize = 0;

    for (uint i = 0; i < m_numGridCells; i++) {
        if (m_hCellStart[i] != 0xffffffff) {
            uint cellSize = m_hCellEnd[i] - m_hCellStart[i];

            //            printf("cell: %d, %d particles\n", i, cellSize);
            if (cellSize > maxCellSize) {
                maxCellSize = cellSize;
            }
        }
    }

    printf("maximum particles per cell = %d\n", maxCellSize);
}

// Downloads position and velocity for count particles starting at start and prints them
// to stdout; a debugging aid for inspecting simulation state.
void ParticleSystem::dumpParticles(uint start, uint count)
{
    // debug
    copyArrayFromDevice(m_hPos, 0, &m_cuda_posvbo_resource, sizeof(float) * 4 * count);
    copyArrayFromDevice(m_hVel, m_dVel, 0, sizeof(float) * 4 * count);

    for (uint i = start; i < start + count; i++) {
        //        printf("%d: ", i);
        printf("pos: (%.4f, %.4f, %.4f, %.4f)\n",
               m_hPos[i * 4 + 0],
               m_hPos[i * 4 + 1],
               m_hPos[i * 4 + 2],
               m_hPos[i * 4 + 3]);
        printf("vel: (%.4f, %.4f, %.4f, %.4f)\n",
               m_hVel[i * 4 + 0],
               m_hVel[i * 4 + 1],
               m_hVel[i * 4 + 2],
               m_hVel[i * 4 + 3]);
    }
}

// Copies the requested device array (POSITION or VELOCITY) into its matching host buffer
// and returns a pointer to that host buffer.
float *ParticleSystem::getArray(ParticleArray array)
{
    assert(m_bInitialized);

    float                       *hdata             = 0;
    float                       *ddata             = 0;
    struct cudaGraphicsResource *cuda_vbo_resource = 0;

    switch (array) {
    default:
    case POSITION:
        hdata             = m_hPos;
        ddata             = m_dPos;
        cuda_vbo_resource = m_cuda_posvbo_resource;
        break;

    case VELOCITY:
        hdata = m_hVel;
        ddata = m_dVel;
        break;
    }

    copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource, m_numParticles * 4 * sizeof(float));
    return hdata;
}

// Uploads count particles' worth of data into the requested device array (POSITION or
// VELOCITY) starting at particle index start; for POSITION in OpenGL mode this goes
// through the VBO rather than a raw CUDA copy.
void ParticleSystem::setArray(ParticleArray array, const float *data, int start, int count)
{
    assert(m_bInitialized);

    switch (array) {
    default:
    case POSITION: {
        if (m_bUseOpenGL) {
            unregisterGLBufferObject(m_cuda_posvbo_resource);
            glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
            glBufferSubData(GL_ARRAY_BUFFER, start * 4 * sizeof(float), count * 4 * sizeof(float), data);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
        }
        else {
            copyArrayToDevice(m_cudaPosVBO, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
        }
    } break;

    case VELOCITY:
        copyArrayToDevice(m_dVel, data, start * 4 * sizeof(float), count * 4 * sizeof(float));
        break;
    }
}

// Returns a pseudo-random float uniformly distributed in [0, 1).
inline float frand() { return rand() / (float)RAND_MAX; }

// Populates the host position/velocity arrays with a regular size[0] x size[1] x size[2]
// lattice of particles (up to numParticles), spaced by spacing and randomly jittered so
// they don't start in a perfectly aligned, degenerate configuration.
void ParticleSystem::initGrid(uint *size, float spacing, float jitter, uint numParticles)
{
    srand(m_seed);

    for (uint z = 0; z < size[2]; z++) {
        for (uint y = 0; y < size[1]; y++) {
            for (uint x = 0; x < size[0]; x++) {
                uint i = (z * size[1] * size[0]) + (y * size[0]) + x;

                if (i < numParticles) {
                    m_hPos[i * 4] = (spacing * x) + m_params.particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
                    m_hPos[i * 4 + 1] =
                        (spacing * y) + m_params.particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
                    m_hPos[i * 4 + 2] =
                        (spacing * z) + m_params.particleRadius - 1.0f + (frand() * 2.0f - 1.0f) * jitter;
                    m_hPos[i * 4 + 3] = 1.0f;

                    m_hVel[i * 4]     = 0.0f;
                    m_hVel[i * 4 + 1] = 0.0f;
                    m_hVel[i * 4 + 2] = 0.0f;
                    m_hVel[i * 4 + 3] = 0.0f;
                }
            }
        }
    }
}

// Reinitializes all particles' positions/velocities on the host according to config
// (a uniform random cloud or a regular grid) and uploads the result to the device.
void ParticleSystem::reset(ParticleConfig config)
{
    switch (config) {
    default:
    case CONFIG_RANDOM: {
        int p = 0, v = 0;

        for (uint i = 0; i < m_numParticles; i++) {
            float point[3];
            point[0]    = frand();
            point[1]    = frand();
            point[2]    = frand();
            m_hPos[p++] = 2 * (point[0] - 0.5f);
            m_hPos[p++] = 2 * (point[1] - 0.5f);
            m_hPos[p++] = 2 * (point[2] - 0.5f);
            m_hPos[p++] = 1.0f; // radius
            m_hVel[v++] = 0.0f;
            m_hVel[v++] = 0.0f;
            m_hVel[v++] = 0.0f;
            m_hVel[v++] = 0.0f;
        }
    } break;

    case CONFIG_GRID: {
        float jitter = m_params.particleRadius * 0.01f;
        uint  s      = (int)ceilf(powf((float)m_numParticles, 1.0f / 3.0f));
        uint  gridSize[3];
        gridSize[0] = gridSize[1] = gridSize[2] = s;
        initGrid(gridSize, m_params.particleRadius * 2.0f, jitter, m_numParticles);
    } break;
    }

    setArray(POSITION, m_hPos, 0, m_numParticles);
    setArray(VELOCITY, m_hVel, 0, m_numParticles);
}

// Fills a solid sphere of radius r cells (spaced by spacing, centered at pos, with all
// particles given initial velocity vel) into the host arrays starting at particle index
// start, then uploads the affected range to the device.
void ParticleSystem::addSphere(int start, float *pos, float *vel, int r, float spacing)
{
    uint index = start;

    for (int z = -r; z <= r; z++) {
        for (int y = -r; y <= r; y++) {
            for (int x = -r; x <= r; x++) {
                float dx     = x * spacing;
                float dy     = y * spacing;
                float dz     = z * spacing;
                float l      = sqrtf(dx * dx + dy * dy + dz * dz);
                float jitter = m_params.particleRadius * 0.01f;

                if ((l <= m_params.particleRadius * 2.0f * r) && (index < m_numParticles)) {
                    m_hPos[index * 4]     = pos[0] + dx + (frand() * 2.0f - 1.0f) * jitter;
                    m_hPos[index * 4 + 1] = pos[1] + dy + (frand() * 2.0f - 1.0f) * jitter;
                    m_hPos[index * 4 + 2] = pos[2] + dz + (frand() * 2.0f - 1.0f) * jitter;
                    m_hPos[index * 4 + 3] = pos[3];

                    m_hVel[index * 4]     = vel[0];
                    m_hVel[index * 4 + 1] = vel[1];
                    m_hVel[index * 4 + 2] = vel[2];
                    m_hVel[index * 4 + 3] = vel[3];
                    index++;
                }
            }
        }
    }

    setArray(POSITION, m_hPos, start, index);
    setArray(VELOCITY, m_hVel, start, index);
}
