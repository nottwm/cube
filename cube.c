/*
 * cube.c - 2D Newtonian Rigid-Body Physics Engine
 * Copyright (c) 2026 NotTWM
 * Licensed under the MIT License (See LICENSE file in root)
 */

#define RAYLIB_GLFW_MODE
#include "raylib.h"
#include "raymath.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <stdarg.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/*
 * SIMD dispatch is decided at COMPILE time from whatever -march the build
 * uses, not guessed at runtime:
 *   - Built with -march=haswell/skylake/native (Kaby Lake and newer, or any
 *     "-march=native" build on such a machine) -> the compiler predefines
 *     __AVX2__ -> we take the 8-floats-at-a-time AVX2 path.
 *   - Any other x86-64 build (default gcc/clang target, older CPU) always
 *     has at least SSE2 (it's part of the baseline x86-64 ABI) -> 4-wide
 *     SSE2 path.
 *   - Non-x86 (ARM etc.) -> plain scalar C, no intrinsics at all.
 */
#if defined(__AVX2__)
    #include <immintrin.h>
    #define CUBE_SIMD_AVX2 1
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
    #include <emmintrin.h>
    #define CUBE_SIMD_SSE2 1
#endif

/* Below this many active cubes, thread wake/sync overhead costs more than
 * the parallel work saves - so small cube counts (like 50) run these loops
 * on the main thread instead of spiking every core just to check a handful
 * of "is this active" flags. */
#define OMP_PARALLEL_THRESHOLD 128

#include "GLFW/glfw3.h"

/*
 * ==============================================================================
 * DEBUG OVERLAY PLATFORM HOOKS (process CPU%, logical core count, GPU name)
 * ==============================================================================
 * Honesty note for future maintainers: there is no portable, dependency-free
 * way to read a numeric "GPU usage %" (the kind nvidia-smi/Task Manager show)
 * from inside the app itself - that needs a vendor SDK (NVML for NVIDIA,
 * ADLX for AMD, etc.), which this file intentionally does not link against.
 * What IS available everywhere: which GPU/driver is actually being used
 * (glGetString is a stable OpenGL 1.1 entry point every desktop GL library
 * exports statically, so this doesn't need any extra headers or linking
 * beyond what raylib's own OpenGL backend already pulls in) and how much
 * CPU time this process is burning (via the OS, platform-gated below).
 */
#if defined(_WIN32)
    #include <windows.h>
#else
    #include <sys/resource.h>
    #include <unistd.h>
#endif

#ifndef GL_VENDOR
    #define GL_VENDOR   0x1F00
    #define GL_RENDERER 0x1F01
#endif
// Deliberately NOT declared as `extern ... glGetString(...)` - that assumes
// it's statically linked into the executable (true on some platforms/build
// setups, false on others depending on how raylib was linked), which is
// exactly the kind of "works on my machine" bug we don't want here. Instead
// we fetch it through GLFW's own loader, which is guaranteed to resolve any
// GL entry point - including old GL 1.1 ones like glGetString - once a
// context is current, the same mechanism glad/GLEW use under the hood.
typedef const unsigned char *(*PFNGLGETSTRINGPROC)(unsigned int name);

static int GetLogicalCoreCount(void) {
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (si.dwNumberOfProcessors > 0) ? (int)si.dwNumberOfProcessors : 1;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? (int)n : 1;
#endif
}

// Process CPU usage, normalized to 0-100% across all logical cores (i.e.
// "half the machine" reads 50%, matching Task Manager's per-process view
// rather than top's default of summing to 100%-per-core). Refreshed at most
// twice a second from raylib's own GetTime() clock so the number is stable
// and readable instead of jittering every frame; cheap OS call either way.
static float GetProcessCPUPercent(void) {
    static double lastWallTime = -1.0;
    static double lastCPUTime = 0.0;
    static float cachedPercent = 0.0f;
    static int coreCount = 0;
    if (coreCount == 0) coreCount = GetLogicalCoreCount();

    double wallNow = GetTime();
    double cpuNow;

#if defined(_WIN32)
    FILETIME creation, exitTime, kernelTime, userTime;
    if (!GetProcessTimes(GetCurrentProcess(), &creation, &exitTime, &kernelTime, &userTime)) {
        return cachedPercent;
    }
    ULARGE_INTEGER k, u;
    k.LowPart = kernelTime.dwLowDateTime; k.HighPart = kernelTime.dwHighDateTime;
    u.LowPart = userTime.dwLowDateTime;   u.HighPart = userTime.dwHighDateTime;
    cpuNow = (double)(k.QuadPart + u.QuadPart) / 1e7; // 100ns units -> seconds
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return cachedPercent;
    }
    cpuNow = (usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6) +
             (usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6);
#endif

    if (lastWallTime < 0.0) {
        lastWallTime = wallNow;
        lastCPUTime = cpuNow;
        return cachedPercent;
    }

    double wallDelta = wallNow - lastWallTime;
    if (wallDelta >= 0.5) {
        double cpuDelta = cpuNow - lastCPUTime;
        float pct = (float)((cpuDelta / wallDelta) * 100.0 / (double)coreCount);
        cachedPercent = Clamp(pct, 0.0f, 100.0f);
        lastWallTime = wallNow;
        lastCPUTime = cpuNow;
    }

    return cachedPercent;
}

// glGetString only works once a GL context is current, so call this AFTER
// InitWindow(). Resolved once via glfwGetProcAddress (works regardless of
// whether the platform's GL is statically linked) and cached in a static
// buffer - not re-fetched per frame.
static const char *GetGPURendererName(void) {
    static char cached[256] = { 0 };
    static bool fetched = false;
    if (!fetched) {
        fetched = true;
        PFNGLGETSTRINGPROC glGetStringPtr = (PFNGLGETSTRINGPROC)glfwGetProcAddress("glGetString");
        const unsigned char *renderer = glGetStringPtr ? glGetStringPtr(GL_RENDERER) : NULL;
        if (renderer) {
            strncpy(cached, (const char *)renderer, sizeof(cached) - 1);
        } else {
            strncpy(cached, "Unknown", sizeof(cached) - 1);
        }
    }
    return cached;
}

#define APP_VERSION "6.16.0"
#define MAX_CUBES 1000
#define CUBE_SIZE 60.0f
#define MASS 2.0f
#define DEFAULT_GRAVITY 1600.0f

#define MAX_RESTITUTION 0.0f      
#define STATIC_FRICTION 0.65f      
#define KINETIC_FRICTION 0.45f     
#define AIR_RESISTANCE 0.001f      
#define BORDER_MARGIN 5.0f
#define PHYSICS_SUBSTEPS 8         
#define SOLVER_ITERATIONS 10       
#define PENETRATION_SLOP 0.05f     

#define SHOCKWAVE_FORCE 1500000.0f
#define GLOW_RADIUS 70.0f          

#define BASE_DRAG_STIFFNESS 300.0f  
#define BASE_DRAG_DAMPING 20.0f     

/* --- Spawn animation / spawn toss --- */
#define SPAWN_ANIM_DURATION 0.28f
#define SPAWN_TOSS_VX_MIN (-160.0f)
#define SPAWN_TOSS_VX_MAX (160.0f)
#define SPAWN_TOSS_VY_MIN (-260.0f)
#define SPAWN_TOSS_VY_MAX (-40.0f)
#define SPAWN_TOSS_ANGVEL_MAX 6.0f

/* --- Particles (GPU-batched via raylib's immediate-mode draw batching, no CPU rasterization) --- */
#define MAX_PARTICLES 4096
#define PARTICLE_GRAVITY 900.0f
#define PARTICLE_DRAG 0.985f
#define IMPACT_SPEED_THRESHOLD 180.0f

typedef struct {
    Vector2 position;
    Vector2 velocity;
    float rotation;        // Radians
    float angularVelocity; // Rad/sec
    bool isDragged;
    Vector2 localGrabPt;   
    Color baseColor;       
    bool active;
    bool isGrounded;       
    float spawnTimer;      // Counts up from 0 on spawn; drives the pop-in scale animation
} RigidCube;

typedef struct {
    Vector2 position;
    Vector2 velocity;
    float life;
    float maxLife;
    float size;
    Color color;
    bool active;
} Particle;

typedef struct SpatialNode {
    int cubeIndex;
    struct SpatialNode *next;
} SpatialNode;

#define HASH_TABLE_SIZE 1024
static SpatialNode *g_hashTable[HASH_TABLE_SIZE];
static SpatialNode g_nodePool[MAX_CUBES * 4];
static int g_nodePoolCount = 0;
static unsigned int g_pairCheckTokens[MAX_CUBES];
static unsigned int g_tokenCounter = 1;

static bool g_debugMode = false;
static bool g_advancedDebugMode = false;
static bool g_enableCubeCollisions = true;
static bool g_showDebugOverlay = false; // F3

static char g_lastLogBuffer[512] = {0};
static int g_repeatCount = 0;
#define MAX_REPEAT_THROTTLE 10

#define LOG_DEBUG(...) do { \
    if (g_debugMode || g_advancedDebugMode) { \
        printf("[DEBUG] " __VA_ARGS__); \
        printf("\n"); \
    } \
} while(0)

void LogAdvancedDedup(const char *fmt, ...) {
    if (!g_advancedDebugMode) return;

    char currentBuffer[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(currentBuffer, sizeof(currentBuffer), fmt, args);
    va_end(args);

    if (strcmp(currentBuffer, g_lastLogBuffer) == 0) {
        g_repeatCount++;
        if (g_repeatCount == MAX_REPEAT_THROTTLE) {
            printf("[ADV-DEBUG] (Previous message repeated 10 times - throttling log output)\n");
        }
    } else {
        strncpy(g_lastLogBuffer, currentBuffer, sizeof(g_lastLogBuffer) - 1);
        g_repeatCount = 1;
        printf("[ADV-DEBUG] %s\n", currentBuffer);
    }
}

Color GetRandomDynamicColor(void) {
    return (Color){
        (unsigned char)GetRandomValue(50, 255),
        (unsigned char)GetRandomValue(50, 255),
        (unsigned char)GetRandomValue(50, 255),
        255
    };
}

float RandRangeF(float lo, float hi) {
    return lo + (float)GetRandomValue(0, 10000) / 10000.0f * (hi - lo);
}

/*
 * ==============================================================================
 * PARTICLE SYSTEM (GPU-batched)
 * ==============================================================================
 * A flat, fixed-size pool with a ring-buffer cursor - no allocation, no per-frame
 * CPU rasterization. Every particle is just a DrawCircleV call, which raylib
 * batches into a handful of hardware draw calls via its internal vertex buffer,
 * same as the cube glow sprites. Update is O(MAX_PARTICLES) and OpenMP-parallel.
 */
static Particle g_particles[MAX_PARTICLES];
static int g_particleCursor = 0;

void SpawnParticleBurst(Vector2 origin, Color color, int count, float minSpeed, float maxSpeed, float minLife, float maxLife, float minSize, float maxSize) {
    for (int n = 0; n < count; n++) {
        Particle *p = &g_particles[g_particleCursor];
        g_particleCursor = (g_particleCursor + 1) % MAX_PARTICLES;

        float angle = RandRangeF(0.0f, 2.0f * PI);
        float speed = RandRangeF(minSpeed, maxSpeed);

        p->position = origin;
        p->velocity = (Vector2){ cosf(angle) * speed, sinf(angle) * speed };
        p->maxLife = RandRangeF(minLife, maxLife);
        p->life = p->maxLife;
        p->size = RandRangeF(minSize, maxSize);
        p->color = color;
        p->active = true;
    }
}

void UpdateParticles(float dt) {
    #pragma omp parallel for schedule(static, 128)
    for (int i = 0; i < MAX_PARTICLES; i++) {
        Particle *p = &g_particles[i];
        if (!p->active) continue;

        p->life -= dt;
        if (p->life <= 0.0f) {
            p->active = false;
            continue;
        }

        p->velocity.y += PARTICLE_GRAVITY * dt;
        p->velocity = Vector2Scale(p->velocity, PARTICLE_DRAG);
        p->position = Vector2Add(p->position, Vector2Scale(p->velocity, dt));
    }
}

void DrawParticles(void) {
    BeginBlendMode(BLEND_ADDITIVE);
    for (int i = 0; i < MAX_PARTICLES; i++) {
        Particle *p = &g_particles[i];
        if (!p->active) continue;

        float t = p->life / p->maxLife;
        Color c = Fade(p->color, t);
        DrawCircleV(p->position, p->size * t, c);
    }
    EndBlendMode();
}

// Only called while the F3 debug overlay is open (see main loop) - a plain
// O(MAX_PARTICLES) scan is fine for a once-a-frame debug readout, but there's
// no reason to pay it when the overlay is hidden.
int CountActiveParticles(void) {
    int count = 0;
    for (int i = 0; i < MAX_PARTICLES; i++) {
        if (g_particles[i].active) count++;
    }
    return count;
}

float GetMomentOfInertia(float mass, float size) {
    return (1.0f / 6.0f) * mass * (size * size);
}

void GetCubeCorners(Vector2 center, float size, float angleRad, Vector2 corners[4]) {
    float half = size / 2.0f;
    float cosA = cosf(angleRad);
    float sinA = sinf(angleRad);

    Vector2 local[4] = {
        {-half, -half},
        { half, -half},
        { half,  half},
        {-half,  half}
    };

    for (int i = 0; i < 4; i++) {
        corners[i].x = center.x + (local[i].x * cosA - local[i].y * sinA);
        corners[i].y = center.y + (local[i].x * sinA + local[i].y * cosA);
    }
}

bool IsPointInCube(Vector2 p, Vector2 corners[4]) {
    bool inside = false;
    for (int i = 0, j = 3; i < 4; j = i++) {
        if (((corners[i].y > p.y) != (corners[j].y > p.y)) &&
            (p.x < (corners[j].x - corners[i].x) * (p.y - corners[i].y) / (corners[j].y - corners[i].y) + corners[i].x)) {
            inside = !inside;
        }
    }
    return inside;
}

/*
 * Same point-in-quad test, but against a slightly outward-inflated copy of the
 * quad. Used only for contact-manifold generation on flush resting contacts:
 * without this, whether a corner counts as "touching" flickers between frames
 * due to plain floating-point noise, so a stacked cube can randomly get a
 * single off-center contact point one frame and two the next. A single
 * contact applies torque a two-point contact wouldn't, and that phantom
 * torque is what reads as an idle stacked cube "sliding for no reason."
 * Inflating the test quad by a hair makes both corners of a flush edge
 * register reliably, every frame, so the contact count stops flickering.
 */
bool IsPointInCubeMargin(Vector2 p, Vector2 corners[4], float margin) {
    Vector2 centroid = { 0, 0 };
    for (int i = 0; i < 4; i++) {
        centroid.x += corners[i].x;
        centroid.y += corners[i].y;
    }
    centroid.x *= 0.25f;
    centroid.y *= 0.25f;

    Vector2 inflated[4];
    for (int i = 0; i < 4; i++) {
        Vector2 dir = Vector2Subtract(corners[i], centroid);
        float len = Vector2Length(dir);
        if (len > 0.0001f) {
            dir = Vector2Scale(dir, (len + margin) / len);
        }
        inflated[i] = Vector2Add(centroid, dir);
    }
    return IsPointInCube(p, inflated);
}

float Cross2DVec(Vector2 a, Vector2 b) {
    return a.x * b.y - a.y * b.x;
}

Vector2 Cross2DScalar(float w, Vector2 r) {
    return (Vector2){ -w * r.y, w * r.x };
}

void ProjectCorners(Vector2 corners[4], Vector2 axis, float *min, float *max) {
    *min = Vector2DotProduct(corners[0], axis);
    *max = *min;
    for (int i = 1; i < 4; i++) {
        float val = Vector2DotProduct(corners[i], axis);
        if (val < *min) *min = val;
        if (val > *max) *max = val;
    }
}

Vector2 LocalToWorldVec(Vector2 local, float angleRad) {
    float cosA = cosf(angleRad);
    float sinA = sinf(angleRad);
    return (Vector2){
        local.x * cosA - local.y * sinA,
        local.x * sinA + local.y * cosA
    };
}

Vector2 WorldToLocalVec(Vector2 world, float angleRad) {
    float cosA = cosf(-angleRad);
    float sinA = sinf(-angleRad);
    return (Vector2){
        world.x * cosA - world.y * sinA,
        world.x * sinA + world.y * cosA
    };
}

/* 
 * ==============================================================================
 * CUBE-TO-CUBE COLLISION RESOLUTION (SAT + IMPULSE SOLVER)
 * ==============================================================================
 * This whole fucking function is a black-magic ritual. It figures out when two 
 * rotating, crashing squares smash into each other, finds where they touch, 
 * pushes them apart so they don't clip through each other like cheap ghosts, 
 * and calculates bounce impulses and friction. Good luck debugging this shit.
 */
void ResolveCubeToCubeCollision(int idxA, int idxB, RigidCube *a, RigidCube *b, float invMass, float invInertia, float dt, bool applyPositionCorrection) {
    // Quick broad-phase distance check so we don't waste CPU cycles on cubes miles apart
    float distSq = Vector2DistanceSqr(a->position, b->position);
    float maxDist = CUBE_SIZE * 1.5f;
    if (distSq > maxDist * maxDist) return;

    float boundingRadius = CUBE_SIZE * 0.70710678f;
    float centerDistX = fabsf(a->position.x - b->position.x);
    float centerDistY = fabsf(a->position.y - b->position.y);
    if (centerDistX > (boundingRadius * 2.0f) || centerDistY > (boundingRadius * 2.0f)) {
        return;
    }

    Vector2 cornersA[4], cornersB[4];
    GetCubeCorners(a->position, CUBE_SIZE, a->rotation, cornersA);
    GetCubeCorners(b->position, CUBE_SIZE, b->rotation, cornersB);

    // Separating Axis Theorem (SAT): Checking edge normal axes to find overlap
    Vector2 axes[4] = {
        Vector2Normalize(Vector2Subtract(cornersA[1], cornersA[0])),
        Vector2Normalize(Vector2Subtract(cornersA[3], cornersA[0])),
        Vector2Normalize(Vector2Subtract(cornersB[1], cornersB[0])),
        Vector2Normalize(Vector2Subtract(cornersB[3], cornersB[0]))
    };

    float minOverlap = FLT_MAX;
    Vector2 normal = { 0, 0 };

    for (int i = 0; i < 4; i++) {
        Vector2 axis = axes[i];
        float minA, maxA, minB, maxB;
        ProjectCorners(cornersA, axis, &minA, &maxA);
        ProjectCorners(cornersB, axis, &minB, &maxB);

        float overlap = fminf(maxA, maxB) - fmaxf(minA, minB);
        if (overlap <= 0.0f) return; // Found a separating axis, no collision!

        if (overlap < minOverlap) {
            minOverlap = overlap;
            normal = axis;
        }
    }

    Vector2 dirAtoB = Vector2Subtract(b->position, a->position);
    if (Vector2DotProduct(dirAtoB, normal) < 0.0f) {
        normal = Vector2Scale(normal, -1.0f);
    }

    // Position correction to fix sinking/overlapping jitter
    if (applyPositionCorrection) {
        float depth = fmaxf(0.0f, minOverlap - PENETRATION_SLOP);
        a->position = Vector2Subtract(a->position, Vector2Scale(normal, depth * 0.5f));
        b->position = Vector2Add(b->position, Vector2Scale(normal, depth * 0.5f));
    }

    Vector2 contacts[4];
    int contactCount = 0;

    #define CONTACT_MARGIN 1.0f
    for (int i = 0; i < 4; i++) {
        if (IsPointInCubeMargin(cornersA[i], cornersB, CONTACT_MARGIN)) contacts[contactCount++] = cornersA[i];
    }
    for (int i = 0; i < 4; i++) {
        if (IsPointInCubeMargin(cornersB[i], cornersA, CONTACT_MARGIN) && contactCount < 4) contacts[contactCount++] = cornersB[i];
    }

    if (contactCount == 0) {
        contacts[0] = Vector2Scale(Vector2Add(a->position, b->position), 0.5f);
        contactCount = 1;
    }

    // Applying linear and angular impulse physics math based on contact points
    for (int c = 0; c < contactCount; c++) {
        Vector2 pt = contacts[c];
        Vector2 rA = Vector2Subtract(pt, a->position);
        Vector2 rB = Vector2Subtract(pt, b->position);

        Vector2 velA = Vector2Add(a->velocity, Cross2DScalar(a->angularVelocity, rA));
        Vector2 velB = Vector2Add(b->velocity, Cross2DScalar(b->angularVelocity, rB));
        Vector2 relVel = Vector2Subtract(velA, velB);

        float vn = Vector2DotProduct(relVel, normal);
        if (vn <= 0.0f) continue;

        float rACrossN = Cross2DVec(rA, normal);
        float rBCrossN = Cross2DVec(rB, normal);

        float impulseDenom = (invMass * 2.0f) +
                             (rACrossN * rACrossN) * invInertia +
                             (rBCrossN * rBCrossN) * invInertia;

        if (impulseDenom <= 0.0001f) continue;

        float jn = vn / (impulseDenom * (float)contactCount);
        if (jn < 0.0f) jn = 0.0f;

        Vector2 normalImpulse = Vector2Scale(normal, jn);

        a->velocity = Vector2Subtract(a->velocity, Vector2Scale(normalImpulse, invMass));
        a->angularVelocity -= Cross2DVec(rA, normalImpulse) * invInertia;

        b->velocity = Vector2Add(b->velocity, Vector2Scale(normalImpulse, invMass));
        b->angularVelocity += Cross2DVec(rB, normalImpulse) * invInertia;

        if (applyPositionCorrection && vn > IMPACT_SPEED_THRESHOLD) {
            float impactRatio = Clamp(vn / 900.0f, 0.0f, 1.0f);
            Color sparkColor = {
                (unsigned char)((a->baseColor.r + b->baseColor.r) / 2),
                (unsigned char)((a->baseColor.g + b->baseColor.g) / 2),
                (unsigned char)((a->baseColor.b + b->baseColor.b) / 2),
                255
            };
            SpawnParticleBurst(pt, sparkColor, (int)(2 + impactRatio * 6), 20.0f, 60.0f + impactRatio * 220.0f, 0.15f, 0.4f, 1.5f, 3.5f);
        }

        Vector2 tangent = { -normal.y, normal.x };
        Vector2 newRelVel = Vector2Subtract(
            Vector2Add(a->velocity, Cross2DScalar(a->angularVelocity, rA)),
            Vector2Add(b->velocity, Cross2DScalar(b->angularVelocity, rB))
        );

        float vt = Vector2DotProduct(newRelVel, tangent);
        float rACrossT = Cross2DVec(rA, tangent);
        float rBCrossT = Cross2DVec(rB, tangent);

        float frictionDenom = (invMass * 2.0f) +
                              (rACrossT * rACrossT) * invInertia +
                              (rBCrossT * rBCrossT) * invInertia;

        if (frictionDenom > 0.0001f) {
            float jtNeeded = vt / (frictionDenom * (float)contactCount);
            
            float maxStaticFriction = STATIC_FRICTION * jn;
            float maxKineticFriction = KINETIC_FRICTION * jn;

            float jt = 0.0f;
            if (fabsf(jtNeeded) <= maxStaticFriction) {
                jt = jtNeeded;
            } else {
                jt = Clamp(jtNeeded, -maxKineticFriction, maxKineticFriction);
            }

            Vector2 frictionImpulse = Vector2Scale(tangent, jt);

            a->velocity = Vector2Subtract(a->velocity, Vector2Scale(frictionImpulse, invMass));
            a->angularVelocity -= Cross2DVec(rA, frictionImpulse) * invInertia;

            b->velocity = Vector2Add(b->velocity, Vector2Scale(frictionImpulse, invMass));
            b->angularVelocity += Cross2DVec(rB, frictionImpulse) * invInertia;
        }
    }
}

/*
 * ==============================================================================
 * BOUNDARY WALL COLLISION RESOLUTION
 * ==============================================================================
 * Keeps cubes trapped inside the window boundaries. Evaluates corner collisions
 * against screen edges so cubes bounce off walls and floors properly.
 */
void ResolveWallCollision(RigidCube *k, float minX, float maxX, float minY, float maxY, float invMass, float invInertia, float dt, bool applyPositionCorrection) {
    Vector2 corners[4];
    GetCubeCorners(k->position, CUBE_SIZE, k->rotation, corners);

    for (int i = 0; i < 4; i++) {
        Vector2 pt = corners[i];
        Vector2 normal = { 0, 0 };
        float penetration = 0.0f;

        if (pt.x < minX) { normal = (Vector2){ 1, 0 }; penetration = minX - pt.x; }
        else if (pt.x > maxX) { normal = (Vector2){ -1, 0 }; penetration = pt.x - maxX; }
        else if (pt.y < minY) { normal = (Vector2){ 0, 1 }; penetration = minY - pt.y; }
        else if (pt.y > maxY) { normal = (Vector2){ 0, -1 }; penetration = pt.y - maxY; }

        if (penetration > 0.0f) {
            if (applyPositionCorrection) {
                float depth = fmaxf(0.0f, penetration - PENETRATION_SLOP);
                k->position = Vector2Add(k->position, Vector2Scale(normal, depth));
            }

            Vector2 r = Vector2Subtract(pt, k->position);
            Vector2 contactVel = Vector2Add(k->velocity, Cross2DScalar(k->angularVelocity, r));
            float vn = Vector2DotProduct(contactVel, normal);

            float rCrossN = Cross2DVec(r, normal);
            float impulseDenom = invMass + (rCrossN * rCrossN) * invInertia;

            if (vn < 0.0f) {
                float jn = -vn / impulseDenom;
                if (jn < 0.0f) jn = 0.0f;

                Vector2 normalImpulse = Vector2Scale(normal, jn);
                k->velocity = Vector2Add(k->velocity, Vector2Scale(normalImpulse, invMass));
                k->angularVelocity += Cross2DVec(r, normalImpulse) * invInertia;

                if (normal.y < -0.5f) {
                    k->isGrounded = true;
                }

                if (applyPositionCorrection && (-vn) > IMPACT_SPEED_THRESHOLD) {
                    float impactRatio = Clamp((-vn) / 900.0f, 0.0f, 1.0f);
                    SpawnParticleBurst(pt, k->baseColor, (int)(2 + impactRatio * 5), 15.0f, 50.0f + impactRatio * 180.0f, 0.12f, 0.35f, 1.5f, 3.0f);
                }

                Vector2 tangent = { -normal.y, normal.x };
                Vector2 updatedContactVel = Vector2Add(k->velocity, Cross2DScalar(k->angularVelocity, r));
                float vt = Vector2DotProduct(updatedContactVel, tangent);

                float rCrossT = Cross2DVec(r, tangent);
                float frictionDenom = invMass + (rCrossT * rCrossT) * invInertia;

                if (frictionDenom > 0.0001f) {
                    float jtNeeded = -vt / frictionDenom;
                    
                    float maxStaticFriction = STATIC_FRICTION * jn;
                    float maxKineticFriction = KINETIC_FRICTION * jn;

                    float jt = 0.0f;
                    if (fabsf(jtNeeded) <= maxStaticFriction) {
                        jt = jtNeeded;
                    } else {
                        jt = Clamp(jtNeeded, -maxKineticFriction, maxKineticFriction);
                    }

                    Vector2 frictionImpulse = Vector2Scale(tangent, jt);
                    k->velocity = Vector2Add(k->velocity, Vector2Scale(frictionImpulse, invMass));
                    k->angularVelocity += Cross2DVec(r, frictionImpulse) * invInertia;
                }
            }
        }
    }
}

/*
 * ==============================================================================
 * BULK CUBE INTEGRATION (gravity, position/rotation, air drag, ground damping)
 * ==============================================================================
 * This is the one loop in the engine that's a genuinely good SIMD target:
 * fixed amount of straight-line float math per cube, same operations for
 * every cube, no early-outs. (Collision resolution isn't - it's full of
 * per-pair early returns and variable contact counts, so it stays scalar;
 * vectorizing branchy code like that usually makes it slower, not faster.)
 *
 * Cube fields live interleaved in RigidCube (AoS), so we pack the handful of
 * floats this loop needs into flat arrays (SoA), run the vector math, then
 * scatter the results back. Those scratch arrays are static (.bss) - sized
 * once at compile time, zero malloc/free per frame, so there's no allocator
 * overhead riding along with the "optimization".
 */
static float g_ix[MAX_CUBES], g_iy[MAX_CUBES];
static float g_ivx[MAX_CUBES], g_ivy[MAX_CUBES];
static float g_irot[MAX_CUBES], g_iang[MAX_CUBES];
static float g_ispawn[MAX_CUBES], g_iGroundMask[MAX_CUBES];

// Plain scalar version - identical math to the original loop. Used as the
// fallback for non-x86 builds, and to mop up the "count % SIMD width" tail.
static void IntegrateCubesScalarRange(RigidCube *cubes, int start, int end, float dt, float gravity) {
    for (int k = start; k < end; k++) {
        RigidCube *c = &cubes[k];

        if (c->spawnTimer < SPAWN_ANIM_DURATION) {
            c->spawnTimer += dt;
        }

        c->velocity.y += gravity * dt;
        c->position.x += c->velocity.x * dt;
        c->position.y += c->velocity.y * dt;
        c->rotation += c->angularVelocity * dt;

        float angularSign = (c->angularVelocity > 0.0f) ? 1.0f : -1.0f;
        float airTorque = AIR_RESISTANCE * (c->angularVelocity * c->angularVelocity) * angularSign;
        c->angularVelocity -= airTorque * dt;
        c->angularVelocity *= 0.999f;

        if (!c->isDragged && c->isGrounded) {
            if (Vector2LengthSqr(c->velocity) < 12.0f) {
                c->velocity.y = 0.0f;
                c->velocity.x *= 0.85f;
            }
            if (fabsf(c->angularVelocity) < 0.02f) {
                c->angularVelocity = 0.0f;
            }
        }

        c->isGrounded = false;
    }
}

#if defined(CUBE_SIMD_AVX2) || defined(CUBE_SIMD_SSE2)
static void IntegrateCubesSIMD(RigidCube *cubes, int count, float dt, float gravity) {
#if defined(CUBE_SIMD_AVX2)
    const int W = 8;
#else
    const int W = 4;
#endif
    int vecCount = (count / W) * W;

    for (int k = 0; k < vecCount; k++) {
        g_ix[k]  = cubes[k].position.x;
        g_iy[k]  = cubes[k].position.y;
        g_ivx[k] = cubes[k].velocity.x;
        g_ivy[k] = cubes[k].velocity.y;
        g_irot[k]  = cubes[k].rotation;
        g_iang[k]  = cubes[k].angularVelocity;
        g_ispawn[k] = cubes[k].spawnTimer;
        g_iGroundMask[k] = (!cubes[k].isDragged && cubes[k].isGrounded) ? 1.0f : 0.0f;
    }

    for (int k = 0; k < vecCount; k += W) {
#if defined(CUBE_SIMD_AVX2)
        __m256 vx = _mm256_loadu_ps(&g_ivx[k]);
        __m256 vy = _mm256_loadu_ps(&g_ivy[k]);
        __m256 px = _mm256_loadu_ps(&g_ix[k]);
        __m256 py = _mm256_loadu_ps(&g_iy[k]);
        __m256 rot = _mm256_loadu_ps(&g_irot[k]);
        __m256 ang = _mm256_loadu_ps(&g_iang[k]);
        __m256 spawn = _mm256_loadu_ps(&g_ispawn[k]);
        // Loaded as 1.0f/0.0f, not a bitmask - turn it into a proper
        // all-ones/all-zeros compare mask before using it with and/andnot.
        __m256 groundMask = _mm256_cmp_ps(_mm256_loadu_ps(&g_iGroundMask[k]), _mm256_set1_ps(0.5f), _CMP_GT_OQ);

        __m256 dtv = _mm256_set1_ps(dt);
        __m256 zero = _mm256_setzero_ps();

        __m256 spawnLT = _mm256_cmp_ps(spawn, _mm256_set1_ps(SPAWN_ANIM_DURATION), _CMP_LT_OQ);
        spawn = _mm256_blendv_ps(spawn, _mm256_add_ps(spawn, dtv), spawnLT);

        vy = _mm256_add_ps(vy, _mm256_mul_ps(_mm256_set1_ps(gravity), dtv));
        px = _mm256_add_ps(px, _mm256_mul_ps(vx, dtv));
        py = _mm256_add_ps(py, _mm256_mul_ps(vy, dtv));
        rot = _mm256_add_ps(rot, _mm256_mul_ps(ang, dtv));

        __m256 signMask = _mm256_cmp_ps(ang, zero, _CMP_GT_OQ);
        __m256 sign = _mm256_blendv_ps(_mm256_set1_ps(-1.0f), _mm256_set1_ps(1.0f), signMask);
        __m256 airTorque = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(AIR_RESISTANCE), _mm256_mul_ps(ang, ang)), sign);
        ang = _mm256_sub_ps(ang, _mm256_mul_ps(airTorque, dtv));
        ang = _mm256_mul_ps(ang, _mm256_set1_ps(0.999f));

        __m256 speedSq = _mm256_add_ps(_mm256_mul_ps(vx, vx), _mm256_mul_ps(vy, vy));
        __m256 slowEnough = _mm256_cmp_ps(speedSq, _mm256_set1_ps(12.0f), _CMP_LT_OQ);
        __m256 applyLinDamp = _mm256_and_ps(groundMask, slowEnough);
        vy = _mm256_blendv_ps(vy, zero, applyLinDamp);
        vx = _mm256_blendv_ps(vx, _mm256_mul_ps(vx, _mm256_set1_ps(0.85f)), applyLinDamp);

        __m256 absAng = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), ang);
        __m256 applyAngDamp = _mm256_and_ps(groundMask, _mm256_cmp_ps(absAng, _mm256_set1_ps(0.02f), _CMP_LT_OQ));
        ang = _mm256_blendv_ps(ang, zero, applyAngDamp);

        _mm256_storeu_ps(&g_ix[k], px);
        _mm256_storeu_ps(&g_iy[k], py);
        _mm256_storeu_ps(&g_ivx[k], vx);
        _mm256_storeu_ps(&g_ivy[k], vy);
        _mm256_storeu_ps(&g_irot[k], rot);
        _mm256_storeu_ps(&g_iang[k], ang);
        _mm256_storeu_ps(&g_ispawn[k], spawn);
#else
        __m128 vx = _mm_loadu_ps(&g_ivx[k]);
        __m128 vy = _mm_loadu_ps(&g_ivy[k]);
        __m128 px = _mm_loadu_ps(&g_ix[k]);
        __m128 py = _mm_loadu_ps(&g_iy[k]);
        __m128 rot = _mm_loadu_ps(&g_irot[k]);
        __m128 ang = _mm_loadu_ps(&g_iang[k]);
        __m128 spawn = _mm_loadu_ps(&g_ispawn[k]);
        // Loaded as 1.0f/0.0f, not a bitmask - turn it into a proper
        // all-ones/all-zeros compare mask before using it with and/andnot.
        __m128 groundMask = _mm_cmpgt_ps(_mm_loadu_ps(&g_iGroundMask[k]), _mm_set1_ps(0.5f));

        __m128 dtv = _mm_set1_ps(dt);
        __m128 zero = _mm_setzero_ps();

        __m128 spawnLT = _mm_cmplt_ps(spawn, _mm_set1_ps(SPAWN_ANIM_DURATION));
        spawn = _mm_or_ps(_mm_and_ps(spawnLT, _mm_add_ps(spawn, dtv)), _mm_andnot_ps(spawnLT, spawn));

        vy = _mm_add_ps(vy, _mm_mul_ps(_mm_set1_ps(gravity), dtv));
        px = _mm_add_ps(px, _mm_mul_ps(vx, dtv));
        py = _mm_add_ps(py, _mm_mul_ps(vy, dtv));
        rot = _mm_add_ps(rot, _mm_mul_ps(ang, dtv));

        __m128 signMask = _mm_cmpgt_ps(ang, zero);
        __m128 sign = _mm_or_ps(_mm_and_ps(signMask, _mm_set1_ps(1.0f)), _mm_andnot_ps(signMask, _mm_set1_ps(-1.0f)));
        __m128 airTorque = _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(AIR_RESISTANCE), _mm_mul_ps(ang, ang)), sign);
        ang = _mm_sub_ps(ang, _mm_mul_ps(airTorque, dtv));
        ang = _mm_mul_ps(ang, _mm_set1_ps(0.999f));

        __m128 speedSq = _mm_add_ps(_mm_mul_ps(vx, vx), _mm_mul_ps(vy, vy));
        __m128 slowEnough = _mm_cmplt_ps(speedSq, _mm_set1_ps(12.0f));
        __m128 applyLinDamp = _mm_and_ps(groundMask, slowEnough);
        vy = _mm_or_ps(_mm_and_ps(applyLinDamp, zero), _mm_andnot_ps(applyLinDamp, vy));
        __m128 vxScaled = _mm_mul_ps(vx, _mm_set1_ps(0.85f));
        vx = _mm_or_ps(_mm_and_ps(applyLinDamp, vxScaled), _mm_andnot_ps(applyLinDamp, vx));

        __m128 absMask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
        __m128 absAng = _mm_and_ps(ang, absMask);
        __m128 applyAngDamp = _mm_and_ps(groundMask, _mm_cmplt_ps(absAng, _mm_set1_ps(0.02f)));
        ang = _mm_or_ps(_mm_and_ps(applyAngDamp, zero), _mm_andnot_ps(applyAngDamp, ang));

        _mm_storeu_ps(&g_ix[k], px);
        _mm_storeu_ps(&g_iy[k], py);
        _mm_storeu_ps(&g_ivx[k], vx);
        _mm_storeu_ps(&g_ivy[k], vy);
        _mm_storeu_ps(&g_irot[k], rot);
        _mm_storeu_ps(&g_iang[k], ang);
        _mm_storeu_ps(&g_ispawn[k], spawn);
#endif
    }

    for (int k = 0; k < vecCount; k++) {
        cubes[k].position.x = g_ix[k];
        cubes[k].position.y = g_iy[k];
        cubes[k].velocity.x = g_ivx[k];
        cubes[k].velocity.y = g_ivy[k];
        cubes[k].rotation = g_irot[k];
        cubes[k].angularVelocity = g_iang[k];
        cubes[k].spawnTimer = g_ispawn[k];
        cubes[k].isGrounded = false;
    }

    // Remainder that didn't fill a full vector (count % W cubes)
    IntegrateCubesScalarRange(cubes, vecCount, count, dt, gravity);
}
#endif

// Single entry point the main loop calls - picks SIMD vs scalar once here,
// so nothing else in the file needs to know or care which path is active.
static void IntegrateCubes(RigidCube *cubes, int count, float dt, float gravity) {
#if defined(CUBE_SIMD_AVX2) || defined(CUBE_SIMD_SSE2)
    // Packing/unpacking into SoA scratch has fixed overhead per call; below
    // one vector's worth there's nothing to amortize it with, so just do it
    // scalar rather than pay pack/unpack cost to vectorize 1-3 cubes.
    if (count >= 32) {
        IntegrateCubesSIMD(cubes, count, dt, gravity);
        return;
    }
#endif
    IntegrateCubesScalarRange(cubes, 0, count, dt, gravity);
}

/*
 * ==============================================================================
 * SCENE BACKGROUND
 * ==============================================================================
 * Cheap GPU-only draw calls (gradient fill + grid lines), no textures to
 * regenerate, no per-pixel CPU work - raylib batches these into a couple of
 * hardware draw calls per frame, same as the shapes.c grid backdrop.
 */
void DrawSceneBackground(int screenWidth, int screenHeight) {
    Color topColor = (Color){ 5, 10, 15, 255 };
    Color bottomColor = (Color){ 0, 30, 50, 255 };
    DrawRectangleGradientV(0, 0, screenWidth, screenHeight, topColor, bottomColor);

    Color gridColor = (Color){ 255, 255, 255, 10 };
    int gridSpacing = 48;
    for (int x = 0; x < screenWidth; x += gridSpacing) DrawLine(x, 0, x, screenHeight, gridColor);
    for (int y = 0; y < screenHeight; y += gridSpacing) DrawLine(0, y, screenWidth, y, gridColor);
}

void PrintHelp(void) {
    printf("Usage: cube.c [OPTIONS]\n\n");
    printf("cube.c - 2D Newtonian Rigid-Body Physics Engine (Hardware acceleration)\n\n");
    printf("Controls:\n");
    printf("  Left Click + Drag : Dynamic system-scaled drag physics\n");
    printf("  Right Click       : Trigger a radical shockwave explosion\n");
    printf("  N                 : Spawn a new cube at cursor\n");
    printf("  B                 : Toggle inter-cube collisions\n");
    printf("  F3                : Toggle expanded debug overlay (top-right)\n");
    printf("  R                 : Reset scene\n");
    printf("  Spacebar          : Toggle gravity\n");
    printf("  F11               : Toggle borderless fullscreen\n");
    printf("  Hold ESC (3s)     : Quit application\n\n");
    printf("Options:\n");
    printf("  -d, --debug           Enable debug logging\n");
    printf("  -D, --advanced-debug  Enable verbose logging (WARNING: May cause lag!)\n");
    printf("  -v, --version         Display application version\n");
    printf("  -h, --help            Display this help menu\n");
}

// -----------------------------------------------------------------------------
// CAMERA AND SHAKE SYSTEM (ADDED FOR VISUAL EFFECTS)
// -----------------------------------------------------------------------------
static Camera2D g_camera = { 0 };
static float g_shakeIntensity = 0.0f;
static float g_shakeDuration = 0.0f;

void TriggerScreenShake(float intensity, float duration) {
    g_shakeIntensity = intensity;
    g_shakeDuration = duration;
}

void UpdateCameraShake(float dt) {
    if (g_shakeDuration > 0.0f) {
        g_shakeDuration -= dt;
        if (g_shakeDuration < 0.0f) g_shakeDuration = 0.0f;

        // Calculate decayed intensity
        float t = g_shakeDuration / 0.5f; // Normalized based on typical max duration (adjust as needed)
        if (t > 1.0f) t = 1.0f;
        float currentIntensity = g_shakeIntensity * t;

        // Random jitter
        g_camera.target.x = (GetRandomValue(-100, 100) / 100.0f) * currentIntensity;
        g_camera.target.y = (GetRandomValue(-100, 100) / 100.0f) * currentIntensity;
    } else {
        g_camera.target.x = 0.0f;
        g_camera.target.y = 0.0f;
        g_shakeIntensity = 0.0f;
    }
}

int main(int argc, char *argv[]) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            PrintHelp();
            return 0;
        } else if (strcmp(argv[i], "--version") == 0 || strcmp(argv[i], "-v") == 0) {
            printf("######   cube.c, version %s\n", APP_VERSION);
            printf("######   Copyright (c) NotTWM <270893670+totallynotaworldmachine479-source@users.noreply.github.com>\n");
            printf("######   Licensed under the MIT License (See LICENSE or https://opensource.org/license/mit)\n");
            return 0;
        } else if (strcmp(argv[i], "--debug") == 0 || strcmp(argv[i], "-d") == 0) {
            g_debugMode = true;
        } else if (strcmp(argv[i], "--advanced-debug") == 0 || strcmp(argv[i], "-D") == 0) {
            g_advancedDebugMode = true;
            g_debugMode = true;
        } else {
            fprintf(stderr, "Error: Unknown argument '%s'\n", argv[i]);
            PrintHelp();
            return 1;
        }
    }

    if (!g_debugMode && !g_advancedDebugMode) {
        SetTraceLogLevel(LOG_WARNING);
    } else {
        printf("==========================================\n");
        printf("              DEBUG MODE [%s]   \n", g_advancedDebugMode ? "ADVANCED DEBUG MODE" : "STANDARD");
        printf("==========================================\n");
    }

    srand((unsigned int)time(NULL));
    SetRandomSeed((unsigned int)time(NULL));

    glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_WAYLAND);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_VSYNC_HINT | FLAG_MSAA_4X_HINT);
    InitWindow(800, 600, "cube.c");

    SetExitKey(KEY_NULL);

    GLFWwindow *glfwWin = glfwGetCurrentContext();

    // Initialize Camera2D
    g_camera.zoom = 1.0f;
    g_camera.offset = (Vector2){ 0, 0 };
    g_camera.rotation = 0.0f;
    g_camera.target = (Vector2){ 0, 0 };

    int glowTexSize = (int)(CUBE_SIZE + (GLOW_RADIUS * 2.0f));
    Image glowImg = GenImageColor(glowTexSize, glowTexSize, (Color){ 0, 0, 0, 0 });
    Vector2 glowCenter = { (float)glowTexSize / 2.0f, (float)glowTexSize / 2.0f };
    float baseRadius = (CUBE_SIZE / 2.0f);

    for (int y = 0; y < glowTexSize; y++) {
        for (int x = 0; x < glowTexSize; x++) {
            float dist = Vector2Distance((Vector2){ (float)x, (float)y }, glowCenter);
            if (dist >= baseRadius && dist <= (baseRadius + GLOW_RADIUS)) {
                float progress = (dist - baseRadius) / GLOW_RADIUS;
                float softFalloff = powf(1.0f - progress, 2.2f);
                unsigned char alpha = (unsigned char)(255 * 0.06f * softFalloff);
                ImageDrawPixel(&glowImg, x, y, (Color){ 255, 255, 255, alpha });
            }
        }
    }
    Texture2D glowTexture = LoadTextureFromImage(glowImg);
    UnloadImage(glowImg);
    SetTextureFilter(glowTexture, TEXTURE_FILTER_BILINEAR);

    RigidCube cubes[MAX_CUBES] = { 0 };

    Color initialColor = (Color){ 255, 50, 50, 255 };
    cubes[0] = (RigidCube){
        .position = { 400, 200 },
        .velocity = { 0, 0 },
        .rotation = 0.0f,
        .angularVelocity = 0.0f,
        .isDragged = false,
        .baseColor = initialColor,
        .active = true,
        .isGrounded = false,
        .spawnTimer = SPAWN_ANIM_DURATION // Skip pop-in animation for the very first cube
    };
    LOG_DEBUG("SPAWNED INITIAL CUBE #0 with Color RGB(%d, %d, %d)", initialColor.r, initialColor.g, initialColor.b);

    int activeCubeCount = 1;
    int draggedCubeIdx = -1;

    float currentGravity = DEFAULT_GRAVITY;
    float invMass = 1.0f / MASS;
    float inertia = GetMomentOfInertia(MASS, CUBE_SIZE);
    float invInertia = 1.0f / inertia;

    int windowedW = 800;
    int windowedH = 600;
    int lastScreenWidth = 800;
    int lastScreenHeight = 600;

    float quitHoldTimer = 0.0f;
    const float quitHoldRequired = 3.0f;
    unsigned long frameCounter = 0;
    double lastFrameDrawMs = 0.0; // draw-submit time is only known AFTER EndDrawing, so the F3 overlay shows last frame's number (standard one-frame-delayed timing display)

    LOG_DEBUG("Initialized physics engine with hardware acceleration: Mass=%.1f, Inertia=%.2f", MASS, inertia);

    while (!WindowShouldClose()) {
        frameCounter++;
        float frameDt = GetFrameTime();
        if (frameDt > 0.02f) frameDt = 0.02f;

        // Update camera shake each frame
        UpdateCameraShake(frameDt);

        float xScale = 1.0f, yScale = 1.0f;
        if (glfwWin) {
            glfwGetWindowContentScale(glfwWin, &xScale, &yScale);
        }
        
        float systemSensitivityScale = (xScale + yScale) * 0.5f;

        float dragStiffness = BASE_DRAG_STIFFNESS * systemSensitivityScale;
        float dragDamping = BASE_DRAG_DAMPING * systemSensitivityScale;

        if (IsKeyDown(KEY_ESCAPE)) {
            quitHoldTimer += frameDt;
            LogAdvancedDedup("Frame #%lu | Escape Key Down Timer: %.3f / %.3f s", frameCounter, quitHoldTimer, quitHoldRequired);
            if (quitHoldTimer >= quitHoldRequired) {
                LOG_DEBUG("Exiting the application, due to escape key threshold");
                break;
            }
        } else {
            if (quitHoldTimer > 0.0f) {
                quitHoldTimer -= frameDt * 6.0f;
                if (quitHoldTimer < 0.0f) quitHoldTimer = 0.0f;
            }
        }

        if (IsKeyPressed(KEY_F11)) {
            if (!IsWindowFullscreen()) {
                if (glfwWin) glfwGetWindowSize(glfwWin, &windowedW, &windowedH);
                ToggleFullscreen();
                LOG_DEBUG("ENTER FULLSCREEN (Saved windowed size: %dx%d)", windowedW, windowedH);
            } else {
                ToggleFullscreen();
                SetWindowSize(windowedW, windowedH);
                LOG_DEBUG("EXIT FULLSCREEN (Restoring windowed size: %dx%d)", windowedW, windowedH);
            }
        }

        if (IsKeyPressed(KEY_SPACE)) {
            currentGravity = (currentGravity > 0.0f) ? 0.0f : DEFAULT_GRAVITY;
            LOG_DEBUG("GRAVITY TOGGLED -> %.1f", currentGravity);
        }

        if (IsKeyPressed(KEY_B)) {
            g_enableCubeCollisions = !g_enableCubeCollisions;
            LOG_DEBUG("INTER-CUBE COLLISIONS %s", g_enableCubeCollisions ? "ENABLED" : "DISABLED");
        }

        if (IsKeyPressed(KEY_F3)) {
            g_showDebugOverlay = !g_showDebugOverlay;
        }

        Vector2 mousePos = GetMousePosition();

        if (IsKeyPressed(KEY_N)) {
            if (activeCubeCount < MAX_CUBES) {
                for (int i = 0; i < MAX_CUBES; i++) {
                    if (!cubes[i].active) {
                        Color newColor = GetRandomDynamicColor();
                        // Give every new cube a little random toss, just like shapes.c's SpawnRandomAtMouse
                        Vector2 tossVel = { RandRangeF(SPAWN_TOSS_VX_MIN, SPAWN_TOSS_VX_MAX), RandRangeF(SPAWN_TOSS_VY_MIN, SPAWN_TOSS_VY_MAX) };
                        float tossSpin = RandRangeF(-SPAWN_TOSS_ANGVEL_MAX, SPAWN_TOSS_ANGVEL_MAX);
                        cubes[i] = (RigidCube){
                            .position = mousePos,
                            .velocity = tossVel,
                            .rotation = 0.0f,
                            .angularVelocity = tossSpin,
                            .isDragged = false,
                            .baseColor = newColor,
                            .active = true,
                            .isGrounded = false,
                            .spawnTimer = 0.0f
                        };
                        activeCubeCount++;
                        SpawnParticleBurst(mousePos, newColor, 18, 40.0f, 220.0f, 0.25f, 0.55f, 1.5f, 4.0f);
                        LOG_DEBUG("SPAWNED CUBE #%d at [%.1f, %.1f] | Color RGB(%d, %d, %d) | Toss [%.1f, %.1f]", 
                                  i, mousePos.x, mousePos.y, newColor.r, newColor.g, newColor.b, tossVel.x, tossVel.y);
                        break;
                    }
                }
            } else {
                LOG_DEBUG("MAX CUBES REACHED (%d)", MAX_CUBES);
            }
        }

        if (IsKeyPressed(KEY_R)) {
            for (int i = 0; i < MAX_CUBES; i++) {
                cubes[i].active = false;
                cubes[i].isDragged = false;
            }
            Color resetColor = (Color){ 255, 50, 50, 255 };
            cubes[0] = (RigidCube){
                .position = { (float)GetScreenWidth() / 2.0f, (float)GetScreenHeight() / 3.0f },
                .velocity = { 0, 0 },
                .rotation = 0.0f,
                .angularVelocity = 0.0f,
                .isDragged = false,
                .baseColor = resetColor,
                .active = true,
                .isGrounded = false,
                .spawnTimer = 0.0f
            };
            activeCubeCount = 1;
            draggedCubeIdx = -1;
            memset(g_particles, 0, sizeof(g_particles));
            SpawnParticleBurst(cubes[0].position, resetColor, 18, 40.0f, 220.0f, 0.25f, 0.55f, 1.5f, 4.0f);
            LOG_DEBUG("RESET SCENE TO SINGLE CUBE | Color RGB(%d, %d, %d)", resetColor.r, resetColor.g, resetColor.b);
        }

        int screenWidth = GetScreenWidth();
        int screenHeight = GetScreenHeight();
        if (glfwWin) glfwGetWindowSize(glfwWin, &screenWidth, &screenHeight);

        int monitorW = GetMonitorWidth(GetCurrentMonitor());
        int monitorH = GetMonitorHeight(GetCurrentMonitor());

        if (!IsWindowFullscreen() && (screenWidth >= monitorW && screenHeight >= monitorH)) {
            screenWidth = windowedW;
            screenHeight = windowedH;
            SetWindowSize(windowedW, windowedH);
        }

        if (!IsWindowFullscreen() && screenWidth < monitorW && screenHeight < monitorH) {
            windowedW = screenWidth;
            windowedH = screenHeight;
        }

        float minX = BORDER_MARGIN;
        float maxX = screenWidth - BORDER_MARGIN;
        float minY = BORDER_MARGIN;
        float maxY = screenHeight - BORDER_MARGIN;

        if (screenWidth != lastScreenWidth || screenHeight != lastScreenHeight) {
            for (int k = 0; k < activeCubeCount; k++) {
                if (!cubes[k].active) continue;

                float rad = cubes[k].rotation;
                float halfDiag = (CUBE_SIZE / 2.0f) * (fabsf(cosf(rad)) + fabsf(sinf(rad)));

                if (cubes[k].position.x > lastScreenWidth / 2.0f) {
                    float distFromRight = (float)lastScreenWidth - cubes[k].position.x;
                    cubes[k].position.x = (float)screenWidth - distFromRight;
                }
                if (cubes[k].position.y > lastScreenHeight / 2.0f) {
                    float distFromBottom = (float)lastScreenHeight - cubes[k].position.y;
                    cubes[k].position.y = (float)screenHeight - distFromBottom;
                }

                cubes[k].position.x = Clamp(cubes[k].position.x, minX + halfDiag, maxX - halfDiag);
                cubes[k].position.y = Clamp(cubes[k].position.y, minY + halfDiag, maxY - halfDiag);
                cubes[k].velocity = (Vector2){ 0, 0 };
                cubes[k].angularVelocity = 0.0f;
            }

            lastScreenWidth = screenWidth;
            lastScreenHeight = screenHeight;
        }

        if (IsMouseButtonPressed(MOUSE_BUTTON_RIGHT)) {
            LogAdvancedDedup("Frame #%lu | Radial Shockwave Triggered at Cursor [%.1f, %.1f]", frameCounter, mousePos.x, mousePos.y);
            // Trigger screen shake on explosion!
            TriggerScreenShake(12.0f, 0.5f); 
            SpawnParticleBurst(mousePos, (Color){ 255, 210, 120, 255 }, 40, 120.0f, 520.0f, 0.3f, 0.7f, 2.0f, 4.5f);
            for (int k = 0; k < activeCubeCount; k++) {
                if (!cubes[k].active) continue;

                Vector2 delta = Vector2Subtract(cubes[k].position, mousePos);
                float dist = Vector2Length(delta);
                if (dist < 10.0f) dist = 10.0f;

                Vector2 dir = Vector2Scale(delta, 1.0f / dist);
                float impulseMagnitude = SHOCKWAVE_FORCE / (dist * dist + 800.0f);
                
                cubes[k].velocity = Vector2Add(cubes[k].velocity, Vector2Scale(dir, impulseMagnitude * invMass));
                cubes[k].angularVelocity += (dir.x * 12.0f);
                cubes[k].isGrounded = false;
            }
        }

        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            for (int k = activeCubeCount - 1; k >= 0; k--) {
                if (!cubes[k].active) continue;

                Vector2 corners[4];
                GetCubeCorners(cubes[k].position, CUBE_SIZE, cubes[k].rotation, corners);

                if (IsPointInCube(mousePos, corners)) {
                    cubes[k].isDragged = true;
                    cubes[k].isGrounded = false;
                    Vector2 grabOffsetWorld = Vector2Subtract(mousePos, cubes[k].position);
                    cubes[k].localGrabPt = WorldToLocalVec(grabOffsetWorld, cubes[k].rotation);
                    draggedCubeIdx = k;
                    break;
                }
            }
        }

        if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT) && draggedCubeIdx != -1) {
            if (cubes[draggedCubeIdx].active) {
                cubes[draggedCubeIdx].isDragged = false;
            }
            draggedCubeIdx = -1;
        }

        /* 
         * ==============================================================================
         * SUB-STEPPED PHYSICS INTEGRATOR LOOP
         * ==============================================================================
         * Breaking each frame down into multiple micro-steps because physics engines 
         * go completely insane and explode if you try to calculate everything in one 
         * massive jump. This keeps stacking and collisions from tearing holes through reality.
         */
        float dt = frameDt / (float)PHYSICS_SUBSTEPS;
        double physicsStepStart = GetTime();

        for (int sub = 0; sub < PHYSICS_SUBSTEPS; sub++) {
            
            if (draggedCubeIdx != -1 && cubes[draggedCubeIdx].active && cubes[draggedCubeIdx].isDragged) {
                RigidCube *c = &cubes[draggedCubeIdx];

                Vector2 r = LocalToWorldVec(c->localGrabPt, c->rotation);
                Vector2 grabPtWorld = Vector2Add(c->position, r);
                Vector2 grabPtVel = Vector2Add(c->velocity, Cross2DScalar(c->angularVelocity, r));

                Vector2 error = Vector2Subtract(mousePos, grabPtWorld);
                Vector2 force = Vector2Subtract(Vector2Scale(error, dragStiffness), Vector2Scale(grabPtVel, dragDamping));

                c->velocity = Vector2Add(c->velocity, Vector2Scale(force, invMass * dt));
                float torque = Cross2DVec(r, force);
                c->angularVelocity += (torque * invInertia) * dt;
                
                LogAdvancedDedup("Frame #%lu Sub #%d | Dragged Cube #%d Position [%.1f, %.1f] Velocity [%.1f, %.1f]", 
                                 frameCounter, sub, draggedCubeIdx, c->position.x, c->position.y, c->velocity.x, c->velocity.y);
            }

            // Bulk gravity/position/rotation/drag integration for every active
            // cube (SIMD-accelerated, see IntegrateCubes above). Bounded to
            // activeCubeCount instead of MAX_CUBES since active cubes are
            // always packed into slots [0, activeCubeCount) - no reason to
            // touch the other (1000 - activeCubeCount) empty slots at all.
            IntegrateCubes(cubes, activeCubeCount, dt, currentGravity);

            /* 
             * ==========================================================================
             * TOKEN-BASED DE-DUPLICATED SPATIAL HASH GRID
             * ==========================================================================
             * This clusterfuck of an optimization sorts every cube into grid buckets so 
             * we don't have to check every single cube against every other cube (which 
             * would be an O(N^2) nightmare). The token system ensures we don't double-check 
             * the exact same pair of cubes twice in the same pass. What a headache.
             */
            if (g_enableCubeCollisions) {
                memset(g_hashTable, 0, sizeof(g_hashTable));
                g_nodePoolCount = 0;
                g_tokenCounter++;
                if (g_tokenCounter == 0) g_tokenCounter = 1;
                memset(g_pairCheckTokens, 0, (size_t)activeCubeCount * sizeof(g_pairCheckTokens[0]));

                for (int i = 0; i < activeCubeCount; i++) {
                    if (!cubes[i].active) continue;

                    int minCellX = (int)((cubes[i].position.x - CUBE_SIZE * 0.6f) / CUBE_SIZE);
                    int maxCellX = (int)((cubes[i].position.x + CUBE_SIZE * 0.6f) / CUBE_SIZE);
                    int minCellY = (int)((cubes[i].position.y - CUBE_SIZE * 0.6f) / CUBE_SIZE);
                    int maxCellY = (int)((cubes[i].position.y + CUBE_SIZE * 0.6f) / CUBE_SIZE);

                    for (int cx = minCellX; cx <= maxCellX; cx++) {
                        for (int cy = minCellY; cy <= maxCellY; cy++) {
                            unsigned int hashKey = (unsigned int)(abs(cx * 73856093 ^ cy * 19349663)) % HASH_TABLE_SIZE;

                            if (g_nodePoolCount < (MAX_CUBES * 4)) {
                                SpatialNode *node = &g_nodePool[g_nodePoolCount++];
                                node->cubeIndex = i;
                                node->next = g_hashTable[hashKey];
                                g_hashTable[hashKey] = node;
                            }
                        }
                    }
                }

                // --- UNIFIED MULTI-PASS SOLVER LOOP VIA SPATIAL BUCKETS ---
                for (int pass = 0; pass < SOLVER_ITERATIONS; pass++) {
                    bool isFirstPass = (pass == 0);

                    for (int h = 0; h < HASH_TABLE_SIZE; h++) {
                        SpatialNode *nodeA = g_hashTable[h];
                        while (nodeA != NULL) {
                            int i = nodeA->cubeIndex;
                            SpatialNode *nodeB = nodeA->next;
                            while (nodeB != NULL) {
                                int j = nodeB->cubeIndex;
                                if (i != j) {
                                    int minIdx = (i < j) ? i : j;
                                    int maxIdx = (i < j) ? j : i;
                                    
                                    unsigned int pairToken = (unsigned int)(minIdx * 1000 + maxIdx);
                                    if (g_pairCheckTokens[maxIdx] != pairToken) {
                                        g_pairCheckTokens[maxIdx] = pairToken;
                                        ResolveCubeToCubeCollision(minIdx, maxIdx, &cubes[minIdx], &cubes[maxIdx], invMass, invInertia, dt, isFirstPass);
                                        LogAdvancedDedup("Frame #%lu Sub #%d Pass #%d | Resolved Cube Pair (%d, %d)", frameCounter, sub, pass, minIdx, maxIdx);
                                    }
                                }
                                nodeB = nodeB->next;
                            }
                            nodeA = nodeA->next;
                        }
                    }
                }
            }

            // Boundary Wall/Floor Pass
            for (int pass = 0; pass < SOLVER_ITERATIONS; pass++) {
                bool isFirstPass = (pass == 0);
                // "if" clause: only actually spins up worker threads once
                // there's enough cubes to make it worth it (see
                // OMP_PARALLEL_THRESHOLD) - otherwise this runs on the main
                // thread like a normal loop, no thread wake/sync overhead.
                #pragma omp parallel for schedule(static, 64) if(activeCubeCount > OMP_PARALLEL_THRESHOLD)
                for (int k = 0; k < activeCubeCount; k++) {
                    if (!cubes[k].active) continue;
                    ResolveWallCollision(&cubes[k], minX, maxX, minY, maxY, invMass, invInertia, dt, isFirstPass);
                }
            }
        }

        double physicsMs = (GetTime() - physicsStepStart) * 1000.0;

        UpdateParticles(frameDt);

        // --- HARDWARE GPU ACCELERATED BATCH RENDER ---
        double drawStart = GetTime();
        BeginDrawing();
            
            // BEGIN CAMERA MODE (World Space)
            BeginMode2D(g_camera);
            
                DrawSceneBackground(screenWidth, screenHeight);

                // Hover Detection (Find which cube the mouse is over)
                int hoveredCubeIdx = -1;
                if (draggedCubeIdx == -1) { // Only check hover if we aren't currently dragging
                    for (int k = activeCubeCount - 1; k >= 0; k--) {
                        if (!cubes[k].active) continue;
                        Vector2 corners[4];
                        GetCubeCorners(cubes[k].position, CUBE_SIZE, cubes[k].rotation, corners);
                        if (IsPointInCube(mousePos, corners)) {
                            hoveredCubeIdx = k;
                            break;
                        }
                    }
                }

                for (int k = 0; k < activeCubeCount; k++) {
                    if (!cubes[k].active) continue;

                    float speed = Vector2Length(cubes[k].velocity) + fabsf(cubes[k].angularVelocity * 30.0f);
                    float speedRatio = Clamp(speed / 1800.0f, 0.0f, 1.0f);

                    Color base = cubes[k].baseColor;
                    Color renderColor = {
                        (unsigned char)Clamp(base.r + (int)(55.0f * speedRatio), 0, 255),
                        (unsigned char)Clamp(base.g + (int)(20.0f * speedRatio), 0, 255),
                        (unsigned char)Clamp(base.b + (int)(40.0f * speedRatio), 0, 255),
                        255
                    };

                    // Pop-in spawn animation: ease-out overshoot scale from 0 -> 1
                    float spawnT = Clamp(cubes[k].spawnTimer / SPAWN_ANIM_DURATION, 0.0f, 1.0f);
                    float easedT = 1.0f - powf(1.0f - spawnT, 3.0f);
                    float overshoot = sinf(spawnT * PI) * 0.18f;
                    float popScale = easedT + overshoot;
                    float drawSize = CUBE_SIZE * popScale;
                    float glowDrawSize = (float)glowTexSize * popScale;

                    // GPU Hardware Vertex Batching via Raylib/OpenGL VBOs
                    DrawTexturePro(
                        glowTexture,
                        (Rectangle){ 0, 0, (float)glowTexSize, (float)glowTexSize },
                        (Rectangle){ cubes[k].position.x, cubes[k].position.y, glowDrawSize, glowDrawSize },
                        (Vector2){ glowDrawSize / 2.0f, glowDrawSize / 2.0f },
                        cubes[k].rotation * RAD2DEG,
                        base
                    );

                    DrawRectanglePro(
                        (Rectangle){ cubes[k].position.x, cubes[k].position.y, drawSize, drawSize },
                        (Vector2){ drawSize / 2.0f, drawSize / 2.0f },
                        cubes[k].rotation * RAD2DEG,
                        renderColor
                    );

                    // Draw Hover Outline (Rotating black border)
                    if (k == hoveredCubeIdx && !cubes[k].isDragged) {
                        // Determine a larger size for the outline
                        float outlineSize = drawSize + 3.0f;
                        float halfOutline = outlineSize / 2.0f;
                        
                        // Calculate the 4 corners of the rotated outline rectangle
                        float angle = cubes[k].rotation;
                        float cosA = cosf(angle);
                        float sinA = sinf(angle);
                        Vector2 pos = cubes[k].position;
                        
                        // The four corners of the unrotated rectangle (centered on pos)
                        Vector2 corners[4] = {
                            { pos.x - halfOutline, pos.y - halfOutline },
                            { pos.x + halfOutline, pos.y - halfOutline },
                            { pos.x + halfOutline, pos.y + halfOutline },
                            { pos.x - halfOutline, pos.y + halfOutline }
                        };
                        
                        // Rotate them around `pos`
                        Vector2 rotCorners[4];
                        for(int i = 0; i < 4; i++) {
                            // Translate to origin relative to pos
                            Vector2 rel = { corners[i].x - pos.x, corners[i].y - pos.y };
                            // Rotate
                            Vector2 rot = { rel.x * cosA - rel.y * sinA, rel.x * sinA + rel.y * cosA };
                            // Translate back
                            rotCorners[i] = (Vector2){ pos.x + rot.x, pos.y + rot.y };
                        }

                        // Draw rotated lines using the mathematically rotated corners
                        DrawLineEx(rotCorners[0], rotCorners[1], 4.0f, (Color){ 0, 0, 0, 220 });
                        DrawLineEx(rotCorners[1], rotCorners[2], 4.0f, (Color){ 0, 0, 0, 220 }); 
                        DrawLineEx(rotCorners[2], rotCorners[3], 4.0f, (Color){ 0, 0, 0, 220 });
                        DrawLineEx(rotCorners[3], rotCorners[0], 4.0f, (Color){ 0, 0, 0, 220 });
                    }
                }

                DrawParticles();

                // --- DRAG WIRE AND HITBOX ---
                if (draggedCubeIdx != -1 && cubes[draggedCubeIdx].active && cubes[draggedCubeIdx].isDragged) {
                    RigidCube *c = &cubes[draggedCubeIdx];
                    
                    // Calculate the exact world position of the grab point on the cube
                    Vector2 r = LocalToWorldVec(c->localGrabPt, c->rotation);
                    Vector2 grabPtWorld = Vector2Add(c->position, r);

                    // Draw the transparent gray wire from cube center to grab point
                    DrawLineV(c->position, grabPtWorld, (Color){ 180, 180, 190, 100 }); 
                    // Draw a slightly thicker wire from the grab point out to the mouse to show the spring tension
                    DrawLineV(grabPtWorld, mousePos, (Color){ 200, 200, 210, 60 });

                    // Draw the small hitbox circle at the cursor
                    DrawCircleLinesV(mousePos, 6.0f, (Color){ 200, 200, 210, 150 });
                    DrawCircleV(mousePos, 2.0f, (Color){ 255, 255, 255, 200 });
                }

            // END CAMERA MODE (Return to Screen Space for UI)
            EndMode2D();

            int currentFps = GetFPS();
            int currentTps = (frameDt > 0.0f) ? (int)(PHYSICS_SUBSTEPS / frameDt) : 0;

            char metricsBuffer[64];
            snprintf(metricsBuffer, sizeof(metricsBuffer), "FPS: %d | TPS: %d", currentFps, currentTps);

            int fontSz = 18;
            int textW = MeasureText(metricsBuffer, fontSz);
            int marginX = 15;
            int marginY = 15;
            int hudX = screenWidth - textW - marginX;
            int hudY = marginY;

            DrawRectangle(hudX - 8, hudY - 4, textW + 16, fontSz + 8, (Color){ 0, 0, 0, 180 });
            DrawText(metricsBuffer, hudX, hudY, fontSz, (Color){ 0, 255, 128, 240 });

            // --- F3 EXPANDED DEBUG OVERLAY ---
            if (g_showDebugOverlay) {
                int panelFontSz = 16;
                int lineHeight = panelFontSz + 6;
                int panelY = hudY + fontSz + 8 + 10; // sits just below the FPS/TPS badge

                int activeParticles = CountActiveParticles();
                float cpuPercent = GetProcessCPUPercent();

                char lines[12][128];
                int lineCount = 0;

                snprintf(lines[lineCount++], sizeof(lines[0]), "Objects: %d / %d", activeCubeCount, MAX_CUBES);
                snprintf(lines[lineCount++], sizeof(lines[0]), "Particles: %d / %d", activeParticles, MAX_PARTICLES);
                snprintf(lines[lineCount++], sizeof(lines[0]), "Physics: %.2f ms  (%d sub x %d iter)", physicsMs, PHYSICS_SUBSTEPS, SOLVER_ITERATIONS);
                snprintf(lines[lineCount++], sizeof(lines[0]), "Draw submit: %.2f ms", lastFrameDrawMs);
                snprintf(lines[lineCount++], sizeof(lines[0]), "CPU: %.1f%%  (%d cores)", cpuPercent, GetLogicalCoreCount());
                snprintf(lines[lineCount++], sizeof(lines[0]), "GPU: %s", GetGPURendererName());
#ifdef _OPENMP
                snprintf(lines[lineCount++], sizeof(lines[0]), "OpenMP threads: %d (active >%d objs)", omp_get_max_threads(), OMP_PARALLEL_THRESHOLD);
#else
                snprintf(lines[lineCount++], sizeof(lines[0]), "OpenMP: disabled at build time");
#endif
                snprintf(lines[lineCount++], sizeof(lines[0]), "Gravity: %s  |  Collisions: %s", (currentGravity > 0.0f) ? "ON" : "OFF", g_enableCubeCollisions ? "ON" : "OFF");
#if defined(CUBE_SIMD_AVX2)
                snprintf(lines[lineCount++], sizeof(lines[0]), "SIMD: AVX2 (8-wide)");
#elif defined(CUBE_SIMD_SSE2)
                snprintf(lines[lineCount++], sizeof(lines[0]), "SIMD: SSE2 (4-wide)");
#else
                snprintf(lines[lineCount++], sizeof(lines[0]), "SIMD: scalar fallback");
#endif
                snprintf(lines[lineCount++], sizeof(lines[0]), "Resolution: %dx%d", screenWidth, screenHeight);
                snprintf(lines[lineCount++], sizeof(lines[0]), "Uptime: %.1f s  |  Frame #%lu", GetTime(), frameCounter);

                int panelW = 0;
                for (int i = 0; i < lineCount; i++) {
                    int w = MeasureText(lines[i], panelFontSz);
                    if (w > panelW) panelW = w;
                }

                int panelX = screenWidth - panelW - marginX;
                int panelH = lineCount * lineHeight + 8;

                DrawRectangle(panelX - 8, panelY - 4, panelW + 16, panelH, (Color){ 0, 0, 0, 180 });
                for (int i = 0; i < lineCount; i++) {
                    DrawText(lines[i], panelX, panelY + i * lineHeight, panelFontSz, (Color){ 220, 220, 230, 240 });
                }
            }

            if (quitHoldTimer > 0.0f) {
                float alphaProgress = Clamp(quitHoldTimer / quitHoldRequired, 0.0f, 1.0f);
                unsigned char textAlpha = (unsigned char)(alphaProgress * 255.0f);
                
                const char *quitText = "Quitting...";
                int fontSize = 28;
                int textWidth = MeasureText(quitText, fontSize);
                int textX = 20;
                int textY = 20;

                DrawRectangle(textX - 10, textY - 6, textWidth + 20, fontSize + 12, (Color){ 0, 0, 0, (unsigned char)(textAlpha * 0.75f) });
                DrawText(quitText, textX, textY, fontSize, (Color){ 255, 255, 255, textAlpha });
            }

        lastFrameDrawMs = (GetTime() - drawStart) * 1000.0;

        EndDrawing();
    }

    UnloadTexture(glowTexture);
    LOG_DEBUG("Shutting down application.");
    CloseWindow();
    return 0;
}
