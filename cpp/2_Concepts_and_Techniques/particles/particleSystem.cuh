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

// C-linkage wrappers around the CUDA kernels defined in particleSystem_cuda.cu / particles_kernel_impl.cuh.
// See particleSystem_cuda.cu for the implementation and detailed comments on each function.
extern "C"
{
    // Selects and initializes the CUDA device to use, honoring any -device= command-line argument.
    void cudaInit(int argc, char **argv);

    // Allocates a device buffer of size bytes and returns it via devPtr.
    void allocateArray(void **devPtr, int size);
    // Frees a device buffer previously returned by allocateArray.
    void freeArray(void *devPtr);

    // Blocks the host until all previously issued CUDA work has completed.
    void threadSync();

    // Copies size bytes from a device buffer (or a mapped OpenGL VBO if cuda_vbo_resource is given) to host memory.
    void copyArrayFromDevice(void *host, const void *device, struct cudaGraphicsResource **cuda_vbo_resource, int size);
    // Copies size bytes from host memory into a device buffer starting at byte offset offset.
    void copyArrayToDevice(void *device, const void *host, int offset, int size);
    // Registers an OpenGL VBO with CUDA so it can be mapped as a device pointer.
    void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
    // Unregisters a previously registered OpenGL/CUDA interop resource.
    void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
    // Maps a registered OpenGL VBO for CUDA access and returns its device pointer.
    void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
    // Unmaps a previously mapped OpenGL/CUDA interop resource, returning it to OpenGL.
    void  unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);

    // Uploads the simulation parameters struct to constant memory for use by all kernels.
    void setParameters(SimParams *hostParams);

    // Launches the kernel that integrates particle positions/velocities forward by deltaTime
    // (applies gravity/damping and handles simple boundary collisions).
    void integrateSystem(float *pos, float *vel, float deltaTime, uint numParticles);

    // Launches the kernel that computes each particle's spatial grid cell hash and stores its
    // original index, as input to the sort-by-cell step.
    void calcHash(uint *gridParticleHash, uint *gridParticleIndex, float *pos, int numParticles);

    // Launches the kernel that reorders positions/velocities into cell-sorted order and records,
    // for every grid cell, the start and end index of its particles in the sorted arrays.
    void reorderDataAndFindCellStart(uint  *cellStart,
                                     uint  *cellEnd,
                                     float *sortedPos,
                                     float *sortedVel,
                                     uint  *gridParticleHash,
                                     uint  *gridParticleIndex,
                                     float *oldPos,
                                     float *oldVel,
                                     uint   numParticles,
                                     uint   numCells);

    // Launches the kernel that, for each particle, examines its 3x3x3 neighboring grid cells,
    // resolves collisions with nearby particles and the interactive collider sphere, and writes
    // the resulting velocity back in original (unsorted) particle order. *numCollisions is set
    // to the number of particle-particle overlaps detected this frame when the COUNT_COLLISIONS
    // instrumentation is compiled in (see particles_kernel_impl.cuh), or 0 otherwise.
    void collide(float *newVel,
                 float *sortedPos,
                 float *sortedVel,
                 uint  *gridParticleIndex,
                 uint  *cellStart,
                 uint  *cellEnd,
                 uint   numParticles,
                 uint   numCells,
                 unsigned int *numCollisions);

    // Sorts particle indices by grid cell hash (radix sort via Thrust) so particles in the same
    // cell become contiguous.
    void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles);
}
