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

#include "GLFW/glfw3.h"

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
} RigidCube;

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

    for (int i = 0; i < 4; i++) {
        if (IsPointInCube(cornersA[i], cornersB)) contacts[contactCount++] = cornersA[i];
    }
    for (int i = 0; i < 4; i++) {
        if (IsPointInCube(cornersB[i], cornersA) && contactCount < 4) contacts[contactCount++] = cornersB[i];
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

void PrintHelp(void) {
    printf("Usage: cube.c [OPTIONS]\n\n");
    printf("cube.c - 2D Newtonian Rigid-Body Physics Engine (Hardware acceleration)\n\n");
    printf("Controls:\n");
    printf("  Left Click + Drag : Dynamic system-scaled drag physics\n");
    printf("  Right Click       : Trigger a radical shockwave explosion\n");
    printf("  N                 : Spawn a new cube at cursor\n");
    printf("  B                 : Toggle inter-cube collisions\n");
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
        .isGrounded = false
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

    LOG_DEBUG("Initialized physics engine with hardware acceleration: Mass=%.1f, Inertia=%.2f", MASS, inertia);

    while (!WindowShouldClose()) {
        frameCounter++;
        float frameDt = GetFrameTime();
        if (frameDt > 0.02f) frameDt = 0.02f;

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

        Vector2 mousePos = GetMousePosition();

        if (IsKeyPressed(KEY_N)) {
            if (activeCubeCount < MAX_CUBES) {
                for (int i = 0; i < MAX_CUBES; i++) {
                    if (!cubes[i].active) {
                        Color newColor = GetRandomDynamicColor();
                        cubes[i] = (RigidCube){
                            .position = mousePos,
                            .velocity = { 0, 0 },
                            .rotation = 0.0f,
                            .angularVelocity = 0.0f,
                            .isDragged = false,
                            .baseColor = newColor,
                            .active = true,
                            .isGrounded = false
                        };
                        activeCubeCount++;
                        LOG_DEBUG("SPAWNED CUBE #%d at [%.1f, %.1f] | Color RGB(%d, %d, %d)", 
                                  i, mousePos.x, mousePos.y, newColor.r, newColor.g, newColor.b);
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
                .isGrounded = false
            };
            activeCubeCount = 1;
            draggedCubeIdx = -1;
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
            for (int k = 0; k < MAX_CUBES; k++) {
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
            for (int k = 0; k < MAX_CUBES; k++) {
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
            for (int k = MAX_CUBES - 1; k >= 0; k--) {
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

            // CPU Multi-Core Parallel Integration via OpenMP
            #pragma omp parallel for schedule(static, 64)
            for (int k = 0; k < MAX_CUBES; k++) {
                if (!cubes[k].active) continue;

                cubes[k].velocity.y += currentGravity * dt;
                cubes[k].position.x += cubes[k].velocity.x * dt;
                cubes[k].position.y += cubes[k].velocity.y * dt;

                cubes[k].rotation += cubes[k].angularVelocity * dt;

                float angularSign = (cubes[k].angularVelocity > 0.0f) ? 1.0f : -1.0f;
                float airTorque = AIR_RESISTANCE * (cubes[k].angularVelocity * cubes[k].angularVelocity) * angularSign;
                cubes[k].angularVelocity -= airTorque * dt;
                cubes[k].angularVelocity *= 0.999f;

                if (!cubes[k].isDragged && cubes[k].isGrounded) {
                    if (Vector2LengthSqr(cubes[k].velocity) < 12.0f) {
                        cubes[k].velocity.y = 0.0f;
                        cubes[k].velocity.x *= 0.85f;
                    }
                    if (fabsf(cubes[k].angularVelocity) < 0.02f) {
                        cubes[k].angularVelocity = 0.0f;
                    }
                }

                cubes[k].isGrounded = false; 
            }

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
                memset(g_pairCheckTokens, 0, sizeof(g_pairCheckTokens));

                for (int i = 0; i < MAX_CUBES; i++) {
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
                #pragma omp parallel for schedule(static, 64)
                for (int k = 0; k < MAX_CUBES; k++) {
                    if (!cubes[k].active) continue;
                    ResolveWallCollision(&cubes[k], minX, maxX, minY, maxY, invMass, invInertia, dt, isFirstPass);
                }
            }
        }

        // --- HARDWARE GPU ACCELERATED BATCH RENDER ---
        BeginDrawing();
            ClearBackground(BLACK);

            for (int k = 0; k < MAX_CUBES; k++) {
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

                // GPU Hardware Vertex Batching via Raylib/OpenGL VBOs
                DrawTexturePro(
                    glowTexture,
                    (Rectangle){ 0, 0, (float)glowTexSize, (float)glowTexSize },
                    (Rectangle){ cubes[k].position.x, cubes[k].position.y, (float)glowTexSize, (float)glowTexSize },
                    (Vector2){ (float)glowTexSize / 2.0f, (float)glowTexSize / 2.0f },
                    cubes[k].rotation * RAD2DEG,
                    base
                );

                DrawRectanglePro(
                    (Rectangle){ cubes[k].position.x, cubes[k].position.y, CUBE_SIZE, CUBE_SIZE },
                    (Vector2){ CUBE_SIZE / 2.0f, CUBE_SIZE / 2.0f },
                    cubes[k].rotation * RAD2DEG,
                    renderColor
                );
            }

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

        EndDrawing();
    }

    UnloadTexture(glowTexture);
    LOG_DEBUG("Shutting down application.");
    CloseWindow();
    return 0;
}
