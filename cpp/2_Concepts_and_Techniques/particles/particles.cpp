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
    Particle system example with collisions using uniform grid

    CUDA 2.1 SDK release 12/2008
    - removed atomic grid method, some optimization, added demo mode.

    CUDA 2.2 release 3/2009
    - replaced sort function with latest radix sort, now disables v-sync.
    - added support for automated testing and comparison to a reference value.
*/

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined(WIN32)
#include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif

// CUDA runtime
#include <cuda_runtime.h>

// CUDA utilities and system includes
#include <helper_cuda.h> // includes cuda.h and cuda_runtime_api.h
#include <helper_functions.h>

// Includes
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdlib.h>

#include "paramgl.h"
#include "particleSystem.h"
#include "render_particles.h"

// Ensure CUDA vector types are available for headers that depend on them
#include <vector_types.h>

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f

#define GRID_SIZE     64
#define NUM_PARTICLES 16384

const uint width = 640, height = 480;

// view params
int                           ox, oy;
int                           buttonState        = 0;
float                         camera_trans[]     = {0, 0, -3};
float                         camera_rot[]       = {0, 0, 0};
float                         camera_trans_lag[] = {0, 0, -3};
float                         camera_rot_lag[]   = {0, 0, 0};
const float                   inertia            = 0.1f;
ParticleRenderer::DisplayMode displayMode        = ParticleRenderer::PARTICLE_SPHERES;

int       mode           = 0;
bool      displayEnabled = true;
bool      bPause         = false;
bool      displaySliders = false;
bool      wireframe      = false;
bool      demoMode       = false;
int       idleCounter    = 0;
int       demoCounter    = 0;
const int idleDelay      = 2000;

enum { M_VIEW = 0, M_MOVE };

uint  numParticles = 0;
uint3 gridSize;
int   numIterations = 0; // run until exit

// simulation parameters
float timestep   = 0.5f;
float damping    = 1.0f;
float gravity    = 0.0003f;
int   iterations = 1;
int   ballr      = 10;

float collideSpring = 0.5f;
;
float collideDamping = 0.02f;
;
float collideShear      = 0.1f;
float collideAttraction = 0.0f;

ParticleSystem *psystem = 0;

// fps
static int          fpsCount = 0;
static int          fpsLimit = 1;
StopWatchInterface *timer    = NULL;

ParticleRenderer *renderer = 0;

float modelView[16];

ParamListGL *params;

// Auto-Verification Code
const int    frameCheckNumber = 4;
unsigned int frameCount       = 0;
unsigned int g_TotalErrors    = 0;
char        *g_refFile        = NULL;

// free-form label written into the benchmark CSV to identify the current configuration
// (e.g. "baseline", "atomic", "shared_mem"); set via --config=<name>
char *g_configLabel = NULL;

// RNG seed used for particle initialization (grid jitter / random cloud); set via --seed=<int>
int g_seed = 1973;


const char *sSDKsample = "CUDA Particles Simulation";

extern "C" void cudaInit(int argc, char **argv);
extern "C" void cudaGLInit(int argc, char **argv);
extern "C" void copyArrayFromDevice(void *host, const void *device, unsigned int vbo, int size);

// Creates the global ParticleSystem (and, if using OpenGL, its ParticleRenderer), resets it
// to a regular grid layout, and starts the FPS timer.
void initParticleSystem(int numParticles, uint3 gridSize, bool bUseOpenGL)
{
    psystem = new ParticleSystem(numParticles, gridSize, bUseOpenGL);
    psystem->setSeed(g_seed);
    psystem->reset(ParticleSystem::CONFIG_GRID);

    if (bUseOpenGL) {
        renderer = new ParticleRenderer;
        renderer->setParticleRadius(psystem->getParticleRadius());
        renderer->setColorBuffer(psystem->getColorBuffer());
    }

    sdkCreateTimer(&timer);
}

// GLUT close callback: destroys the FPS timer and the global ParticleSystem.
void cleanup()
{
    sdkDeleteTimer(&timer);

    if (psystem) {
        delete psystem;
    }
    return;
}

// Creates the GLUT window, verifies the required OpenGL version/extensions are available
// (exiting if not), disables v-sync on Windows, and sets up basic GL state.
void initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA Particles");

    if (!isGLVersionSupported(2, 0) || !areGLExtensionsSupported("GL_ARB_multitexture GL_ARB_vertex_buffer_object")) {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(EXIT_FAILURE);
    }

#if defined(WIN32)

    if (wglewIsSupported("WGL_EXT_swap_control")) {
        // disable vertical sync
        wglSwapIntervalEXT(0);
    }

#endif

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.25, 0.25, 0.25, 1.0);

    glutReportErrors();
}

// Runs the simulation for the given number of iterations with no rendering, timing the run
// to report throughput, and (if a reference file was supplied via -file=) dumps the final
// positions and compares them against that reference to validate correctness.
void runBenchmark(int iterations, char *exec_path)
{
    printf("Run %u particles simulation for %d iterations...\n\n", numParticles, iterations);
    cudaDeviceSynchronize();
    sdkStartTimer(&timer);

    for (int i = 0; i < iterations; ++i) {
        psystem->update(timestep);
    }

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    float fAvgSeconds = ((float)1.0e-3 * (float)sdkGetTimerValue(&timer) / (float)iterations);

    printf("particles, Throughput = %.4f KParticles/s, Time = %.5f s, Size = %u "
           "particles, NumDevsUsed = %u, Workgroup = %u\n",
           (1.0e-3 * numParticles) / fAvgSeconds,
           fAvgSeconds,
           numParticles,
           1,
           0);

    psystem->writeTimingCSV("results/raw/particle_phase_timings.csv", g_configLabel ? g_configLabel : "");

    if (g_refFile) {
        printf("\nChecking result...\n\n");
        float *hPos = (float *)malloc(sizeof(float) * 4 * psystem->getNumParticles());
        copyArrayFromDevice(hPos, psystem->getCudaPosVBO(), 0, sizeof(float) * 4 * psystem->getNumParticles());

        sdkDumpBin((void *)hPos, sizeof(float) * 4 * psystem->getNumParticles(), "particles.bin");

        if (!sdkCompareBin2BinFloat(
                "particles.bin", g_refFile, 4 * psystem->getNumParticles(), MAX_EPSILON_ERROR, THRESHOLD, exec_path)) {
            g_TotalErrors++;
        }
    }
}

// Updates the frame counter and, once fpsLimit frames have elapsed, recomputes and displays
// the current frames-per-second in the window title.
void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit) {
        char  fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "CUDA Particles (%d particles): %3.1f fps", numParticles, ifps);

        glutSetWindowTitle(fps);
        fpsCount = 0;

        fpsLimit = (int)MAX(ifps, 1.f);
        sdkResetTimer(&timer);
    }
}

// GLUT display callback: pushes current UI-controlled parameters into the simulation,
// steps it forward by one timestep (unless paused), then renders the domain wireframe cube,
// the collider sphere, the particles themselves, and (if enabled) the parameter sliders.
void display()
{
    sdkStartTimer(&timer);

    // update the simulation
    if (!bPause) {
        psystem->setIterations(iterations);
        psystem->setDamping(damping);
        psystem->setGravity(-gravity);
        psystem->setCollideSpring(collideSpring);
        psystem->setCollideDamping(collideDamping);
        psystem->setCollideShear(collideShear);
        psystem->setCollideAttraction(collideAttraction);

        psystem->update(timestep);

        if (renderer) {
            renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumParticles());
        }
    }

    // render
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // view transform
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    for (int c = 0; c < 3; ++c) {
        camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
        camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
    }

    glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
    glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
    glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);

    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);

    // cube
    glColor3f(1.0, 1.0, 1.0);
    glutWireCube(2.0);

    // collider
    glPushMatrix();
    float3 p = psystem->getColliderPos();
    glTranslatef(p.x, p.y, p.z);
    glColor3f(1.0, 0.0, 0.0);
    glutSolidSphere(psystem->getColliderRadius(), 20, 10);
    glPopMatrix();

    if (renderer && displayEnabled) {
        renderer->display(displayMode);
    }

    if (displaySliders) {
        glDisable(GL_DEPTH_TEST);
        glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO); // invert color
        glEnable(GL_BLEND);
        params->Render(0, 0);
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
    }

    sdkStopTimer(&timer);

    glutSwapBuffers();
    glutReportErrors();

    computeFPS();
}

// Returns a pseudo-random float uniformly distributed in [0, 1).
inline float frand() { return rand() / (float)RAND_MAX; }

// Injects a new sphere of ballr-radius particles at a random position near the top of the
// simulation domain, with zero initial velocity.
void addSphere()
{
    // inject a sphere of particles
    float pr = psystem->getParticleRadius();
    float tr = pr + (pr * 2.0f) * ballr;
    float pos[4], vel[4];
    pos[0] = -1.0f + tr + frand() * (2.0f - tr * 2.0f);
    pos[1] = 1.0f - tr;
    pos[2] = -1.0f + tr + frand() * (2.0f - tr * 2.0f);
    pos[3] = 0.0f;
    vel[0] = vel[1] = vel[2] = vel[3] = 0.0f;
    psystem->addSphere(0, pos, vel, ballr, pr * 2.0f);
}

// GLUT reshape callback: rebuilds the perspective projection and viewport for the new window
// size and informs the renderer of the new size/FOV (used to scale point-sprite size).
void reshape(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float)w / (float)h, 0.1, 100.0);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);

    if (renderer) {
        renderer->setWindowSize(w, h);
        renderer->setFOV(60.0);
    }
}

// GLUT mouse-button callback: tracks which button(s) are held (adjusted for shift/ctrl
// modifiers) and forwards the event to the parameter-slider UI if it's visible.
void mouse(int button, int state, int x, int y)
{
    int mods;

    if (state == GLUT_DOWN) {
        buttonState |= 1 << button;
    }
    else if (state == GLUT_UP) {
        buttonState = 0;
    }

    mods = glutGetModifiers();

    if (mods & GLUT_ACTIVE_SHIFT) {
        buttonState = 2;
    }
    else if (mods & GLUT_ACTIVE_CTRL) {
        buttonState = 3;
    }

    ox = x;
    oy = y;

    demoMode    = false;
    idleCounter = 0;

    if (displaySliders) {
        if (params->Mouse(x, y, button, state)) {
            glutPostRedisplay();
            return;
        }
    }

    glutPostRedisplay();
}

// Transforms vector v by 4x4 matrix m (column-major, OpenGL convention), writing the result to r.
void xform(float *v, float *r, GLfloat *m)
{
    r[0] = v[0] * m[0] + v[1] * m[4] + v[2] * m[8] + m[12];
    r[1] = v[0] * m[1] + v[1] * m[5] + v[2] * m[9] + m[13];
    r[2] = v[0] * m[2] + v[1] * m[6] + v[2] * m[10] + m[14];
}

// Transforms direction vector v by the transpose (inverse rotation, for an orthonormal
// matrix) of the upper 3x3 of m, writing the result to r; used to map a screen-space drag
// direction into world space.
void ixform(float *v, float *r, GLfloat *m)
{
    r[0] = v[0] * m[0] + v[1] * m[1] + v[2] * m[2];
    r[1] = v[0] * m[4] + v[1] * m[5] + v[2] * m[6];
    r[2] = v[0] * m[8] + v[1] * m[9] + v[2] * m[10];
}

// Transforms world-space point v by the inverse of view matrix m (translates by -m's
// translation, then applies ixform's inverse rotation), writing the result to r.
void ixformPoint(float *v, float *r, GLfloat *m)
{
    float x[4];
    x[0] = v[0] - m[12];
    x[1] = v[1] - m[13];
    x[2] = v[2] - m[14];
    x[3] = 1.0f;
    ixform(x, r, m);
}

// GLUT mouse-motion callback: in M_VIEW mode, updates camera rotation/translation/zoom from
// the drag delta and button state; in M_MOVE mode, drags the interactive collider sphere
// through the scene instead. Also forwards drags to the slider UI when it is visible.
void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if (displaySliders) {
        if (params->Motion(x, y)) {
            ox = x;
            oy = y;
            glutPostRedisplay();
            return;
        }
    }

    switch (mode) {
    case M_VIEW:
        if (buttonState == 3) {
            // left+middle = zoom
            camera_trans[2] += (dy / 100.0f) * 0.5f * fabs(camera_trans[2]);
        }
        else if (buttonState & 2) {
            // middle = translate
            camera_trans[0] += dx / 100.0f;
            camera_trans[1] -= dy / 100.0f;
        }
        else if (buttonState & 1) {
            // left = rotate
            camera_rot[0] += dy / 5.0f;
            camera_rot[1] += dx / 5.0f;
        }

        break;

    case M_MOVE: {
        float  translateSpeed = 0.003f;
        float3 p              = psystem->getColliderPos();

        if (buttonState == 1) {
            float v[3], r[3];
            v[0] = dx * translateSpeed;
            v[1] = -dy * translateSpeed;
            v[2] = 0.0f;
            ixform(v, r, modelView);
            p.x += r[0];
            p.y += r[1];
            p.z += r[2];
        }
        else if (buttonState == 2) {
            float v[3], r[3];
            v[0] = 0.0f;
            v[1] = 0.0f;
            v[2] = dy * translateSpeed;
            ixform(v, r, modelView);
            p.x += r[0];
            p.y += r[1];
            p.z += r[2];
        }

        psystem->setColliderPos(p);
    } break;
    }

    ox = x;
    oy = y;

    demoMode    = false;
    idleCounter = 0;

    glutPostRedisplay();
}

// GLUT keyboard callback: dispatches single-key shortcuts for pausing/stepping the
// simulation, switching camera/collider-move mode, toggling point vs. sphere rendering,
// resetting to a grid/random layout, injecting spheres, dumping debug info, and quitting.
// commented out to remove unused parameter warnings in Linux
void key(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key) {
    case ' ':
        bPause = !bPause;
        break;

    case 13:
        psystem->update(timestep);

        if (renderer) {
            renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getNumParticles());
        }

        break;

    case '\033':
    case 'q':
#if defined(__APPLE__) || defined(MACOSX)
        exit(EXIT_SUCCESS);
#else
        glutDestroyWindow(glutGetWindow());
        return;
#endif
    case 'v':
        mode = M_VIEW;
        break;

    case 'm':
        mode = M_MOVE;
        break;

    case 'p':
        displayMode = (ParticleRenderer::DisplayMode)((displayMode + 1) % ParticleRenderer::PARTICLE_NUM_MODES);
        break;

    case 'd':
        psystem->dumpGrid();
        break;

    case 'u':
        psystem->dumpParticles(0, numParticles - 1);
        break;

    case 'r':
        displayEnabled = !displayEnabled;
        break;

    case '1':
        psystem->reset(ParticleSystem::CONFIG_GRID);
        break;

    case '2':
        psystem->reset(ParticleSystem::CONFIG_RANDOM);
        break;

    case '3':
        addSphere();
        break;

    case '4': {
        // shoot ball from camera
        float pr = psystem->getParticleRadius();
        float vel[4], velw[4], pos[4], posw[4];
        vel[0] = 0.0f;
        vel[1] = 0.0f;
        vel[2] = -0.05f;
        vel[3] = 0.0f;
        ixform(vel, velw, modelView);

        pos[0] = 0.0f;
        pos[1] = 0.0f;
        pos[2] = -2.5f;
        pos[3] = 1.0;
        ixformPoint(pos, posw, modelView);
        posw[3] = 0.0f;

        psystem->addSphere(0, posw, velw, ballr, pr * 2.0f);
    } break;

    case 'w':
        wireframe = !wireframe;
        break;

    case 'h':
        displaySliders = !displaySliders;
        break;
    }

    demoMode    = false;
    idleCounter = 0;
    glutPostRedisplay();
}

// GLUT special-key (arrow/function key) callback: forwards the event to the parameter-slider
// UI when it is visible, and cancels demo mode.
void special(int k, int x, int y)
{
    if (displaySliders) {
        params->Special(k, x, y);
    }

    demoMode    = false;
    idleCounter = 0;
}

// GLUT idle callback: after a period of user inactivity, enters an automatic "demo mode"
// that spins the camera and periodically injects new spheres of particles; always requests
// a redraw of the next frame.
void idle(void)
{
    if ((idleCounter++ > idleDelay) && (demoMode == false)) {
        demoMode = true;
        printf("Entering demo mode\n");
    }

    if (demoMode) {
        camera_rot[1] += 0.1f;

        if (demoCounter++ > 1000) {
            ballr = 10 + (rand() % 10);
            addSphere();
            demoCounter = 0;
        }
    }

    glutPostRedisplay();
}

// When validating against a reference file, zeroes out all randomness/forces for a
// deterministic run; otherwise builds the on-screen ParamListGL slider UI bound to the
// simulation's tunable parameters (timestep, damping, gravity, collision coefficients, etc).
void initParams()
{
    if (g_refFile) {
        timestep          = 0.0f;
        damping           = 0.0f;
        gravity           = 0.0f;
        ballr             = 1;
        collideSpring     = 0.0f;
        collideDamping    = 0.0f;
        collideShear      = 0.0f;
        collideAttraction = 0.0f;
    }
    else {
        // create a new parameter list
        params = new ParamListGL("misc");
        params->AddParam(new Param<float>("time step", timestep, 0.0f, 1.0f, 0.01f, &timestep));
        params->AddParam(new Param<float>("damping", damping, 0.0f, 1.0f, 0.001f, &damping));
        params->AddParam(new Param<float>("gravity", gravity, 0.0f, 0.001f, 0.0001f, &gravity));
        params->AddParam(new Param<int>("ball radius", ballr, 1, 20, 1, &ballr));

        params->AddParam(new Param<float>("collide spring", collideSpring, 0.0f, 1.0f, 0.001f, &collideSpring));
        params->AddParam(new Param<float>("collide damping", collideDamping, 0.0f, 0.1f, 0.001f, &collideDamping));
        params->AddParam(new Param<float>("collide shear", collideShear, 0.0f, 0.1f, 0.001f, &collideShear));
        params->AddParam(
            new Param<float>("collide attract", collideAttraction, 0.0f, 0.1f, 0.001f, &collideAttraction));
    }
}

// Right-click context-menu callback: reuses the keyboard handler so each menu entry behaves
// exactly like pressing its associated key.
void mainMenu(int i) { key((unsigned char)i, 0, 0); }

// Builds the right-click context menu, mapping each entry to the keyboard shortcut it mirrors.
void initMenus()
{
    glutCreateMenu(mainMenu);
    glutAddMenuEntry("Reset block [1]", '1');
    glutAddMenuEntry("Reset random [2]", '2');
    glutAddMenuEntry("Add sphere [3]", '3');
    glutAddMenuEntry("View mode [v]", 'v');
    glutAddMenuEntry("Move cursor mode [m]", 'm');
    glutAddMenuEntry("Toggle point rendering [p]", 'p');
    glutAddMenuEntry("Toggle animation [ ]", ' ');
    glutAddMenuEntry("Step animation [ret]", 13);
    glutAddMenuEntry("Toggle sliders [h]", 'h');
    glutAddMenuEntry("Quit (esc)", '\033');
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
// Entry point: parses command-line options (particle count, grid size, benchmark mode,
// reference-file comparison, iteration count), initializes CUDA (and OpenGL/GLUT unless
// running headless benchmark/validation), creates the particle system, then either runs a
// fixed-iteration benchmark/validation pass or enters the interactive GLUT main loop.
int main(int argc, char **argv)
{
#if defined(__linux__)
    setenv("DISPLAY", ":0", 0);
#endif

    printf("%s Starting...\n\n", sSDKsample);

    printf("NOTE: The CUDA Samples are not meant for performance measurements. "
           "Results may vary when GPU Boost is enabled.\n\n");

    numParticles  = NUM_PARTICLES;
    uint gridDim  = GRID_SIZE;
    numIterations = 0;

    if (argc > 1) {
        if (checkCmdLineFlag(argc, (const char **)argv, "n")) {
            numParticles = getCmdLineArgumentInt(argc, (const char **)argv, "n");
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "grid")) {
            gridDim = getCmdLineArgumentInt(argc, (const char **)argv, "grid");
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "file")) {
            getCmdLineArgumentString(argc, (const char **)argv, "file", &g_refFile);
            fpsLimit      = frameCheckNumber;
            numIterations = 1;
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "config")) {
            getCmdLineArgumentString(argc, (const char **)argv, "config", &g_configLabel);
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "seed")) {
            g_seed = getCmdLineArgumentInt(argc, (const char **)argv, "seed");
        }
    }

    gridSize.x = gridSize.y = gridSize.z = gridDim;
    printf("grid: %d x %d x %d = %d cells\n", gridSize.x, gridSize.y, gridSize.z, gridSize.x * gridSize.y * gridSize.z);
    printf("particles: %d\n", numParticles);
    printf("seed: %d\n", g_seed);

    if (g_configLabel) {
        printf("config: %s\n", g_configLabel);
    }

    bool benchmark = checkCmdLineFlag(argc, (const char **)argv, "benchmark") != 0;

    if (checkCmdLineFlag(argc, (const char **)argv, "i")) {
        numIterations = getCmdLineArgumentInt(argc, (const char **)argv, "i");
    }

    if (benchmark || g_refFile) {
        cudaInit(argc, argv);
    }
    else {
        if (checkCmdLineFlag(argc, (const char **)argv, "device")) {
            printf("[%s]\n", argv[0]);
            printf("   Does not explicitly support -device=n in OpenGL mode\n");
            printf("   To use -device=n, the sample must be running w/o OpenGL\n\n");
            printf(" > %s -device=n -file=<*.bin>\n", argv[0]);
            printf("exiting...\n");
            exit(EXIT_SUCCESS);
        }

        initGL(&argc, argv);
        cudaInit(argc, argv);
    }

    initParticleSystem(numParticles, gridSize, !benchmark && g_refFile == NULL);
    initParams();

    if (benchmark || g_refFile) {
        if (numIterations <= 0) {
            numIterations = 300;
        }

        runBenchmark(numIterations, argv[0]);
    }
    else {
        if (!g_refFile) {
            initMenus();
        }

        glutDisplayFunc(display);
        glutReshapeFunc(reshape);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
        glutKeyboardFunc(key);
        glutSpecialFunc(special);
        glutIdleFunc(idle);

        glutCloseFunc(cleanup);

        glutMainLoop();
    }

    if (psystem) {
        delete psystem;
    }

    exit(g_TotalErrors > 0 ? EXIT_FAILURE : EXIT_SUCCESS);
}
