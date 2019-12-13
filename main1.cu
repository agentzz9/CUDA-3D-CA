/*  

    3D Cellular Automata Simulation,
    Runs on CUDA
    - Sparsh

    OpenGL code referenced from from: https://docs.nvidia.com/cuda/cuda-samples/index.html#simple-opengl

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <helper_gl.h>
#include <GL/freeglut.h>

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#include <vector_types.h>


#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     10 //ms

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width = 1024;
const unsigned int window_height = 1024;

const unsigned int mesh_width = 256;
const unsigned int mesh_height = 256;
const unsigned int mesh_length = 256;


// vbo variables
GLuint vbo;
struct cudaGraphicsResource* cuda_vbo_resource;
void* d_vbo_buffer = NULL;

GLuint loc_vbo;
struct cudaGraphicsResource* cuda_loc_vbo_resource;
void* d_loc_vbo_buffer = NULL;

int* state;
int* state_next;

int* d_state;
int* d_state_next;

float g_fAnim = 0.0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

StopWatchInterface* timer = NULL;

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;

int* pArgc = NULL;
char** pArgv = NULL;

#define MAX(a,b) ((a > b) ? a : b)

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char** argv, char* ref_file);
void cleanup();

// GL functionality
bool initGL(int* argc, char** argv);
void createVBO(GLuint* vbo, struct cudaGraphicsResource** vbo_res,
    unsigned int vbo_res_flags);
void deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);


// Cuda functionality
void runCuda(struct cudaGraphicsResource** vbo_resource, int* state, int* state_next, int* d_state, int* d_state_next);
void runAutoTest(int devID, char** argv, char* ref_file);
void checkResultCuda(int argc, char** argv, const GLuint& vbo);

const char* sSDKsample = "simpleGL (VBO)";

/*
    starting config of the state, input is array of coords you want to spawn live cell
*/
void initializeStartState(int x[], int y[], int z[], int count) {


    // casual fallback on null input, TODO refactor to remove this
    if (x == NULL) {


        for(int x = 0; x < mesh_width; x++)
            for (int y = 0; y < mesh_width; y++)
                for (int z = 0; z < mesh_width; z++)
                {
                    int loc = (mesh_width * (y * mesh_width + x)) + z;
                    state[loc] = 0;
                    state_next[loc] = 0;
                }


        int mid = mesh_width / 2;
        int tx, ty, tz;
        tx = mid;
        ty = mid;
        tz = mid;

        int loc = (mesh_width * (ty * mesh_width + tx)) + tz;


        state[loc] = 1;
       

        tx = mid + 1;
        ty = mid + 1;
        tz = mid + 1;
        loc = (mesh_width * (ty * mesh_width + tx)) + tz;

        state[loc] = 1;


        tx = mid - 1;
        ty = mid - 1;
        tz = mid - 1;
        loc = (mesh_width * (ty * mesh_width + tx)) + tz;

        state[loc] = 1;
        


        /*tx = mid + 1;
        ty = mid;
        tz = mid + 1;
        loc = (mesh_width * (ty * mesh_width + tx)) + tz;

        state[loc] = 1;*/

        /*tx = mid;
        ty = mid + 1;
        tz = mid;
        loc = (mesh_width * (ty * mesh_width + tx)) + tz;

        state[loc] = 1;*/

        

        /*tx = mid + 8;
        ty = mid - 8;
        tz = mid - 8;
        loc = (mesh_width * (ty * mesh_width + tx)) + tz;

        state[loc] = 1;
        
        tx = mid + 8;
        ty = mid - 8;
        tz = mid - 7;
        loc = (mesh_width * (ty * mesh_width + tx)) + tz;

        state[loc] = 1;

        tx = mid + 8;
        ty = mid - 7;
        tz = mid - 8;
        loc = (mesh_width * (ty * mesh_width + tx)) + tz;

        state[loc] = 1;
        */


        return;
    }

    for (int i = 0; i < count; i++) {
        int loc = (mesh_width * (y[i] * mesh_width + x[i])) + z[i];
        state[loc] = 1;
    }

}

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void simple_vbo_kernel(float4* pos, unsigned int length, unsigned int width, unsigned int height, float time, int* state, int* state_next)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    int my_offset = (width * (y * width + x)) + z;
    
    //printf("\nI am %d, %d, %d as %d", x,y,z,my_offset);

    // calculate uvw coordinates
    float u = x / (float)width;
    float v = y / (float)height;
    float w = z / (float)length;
    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;
    w = w * 2.0f - 1.0f;

    //evolution rules
    int count_live = 0, count_dead = 0;
    
    //thank you ms excel
    int dx[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int dy[] = {-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1};
    int dz[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1};

    for (int i = 0; i < 27; i++) {
       
       
        int nx = x + dx[i], ny = y + dy[i], nz = z + dz[i];
        
        if (MAX(MAX(nx, ny), nz) >= width || MIN(MIN(nx, ny), nz) < 0) {
            continue;
        }

        if (nx == x && ny == y && nz == z) {
            continue;
        }
            
        int offset = (width * (ny * width + nx)) + nz;

        if (state[offset] == 1) {
            count_live++;
            //printf("\nvisiting xyz = %d %d %d  neighbor = %d %d %d found alive cell alive count is %d", x, y, z, nx, ny, nz, count_live);
        }else count_dead++;
    
    }

    //basic test rule, spawn new if neighborhood has 2 alive and im empty
    // & write output vertex
    if (state[my_offset] == 0 && count_live == 2) {
        
        //printf("\noffset in cuda: %d, xyz %d %d %d, uvw %f %f %f", my_offset, x, y, z, u, v, w);
        state_next[my_offset] = 1;
        pos[my_offset] = make_float4(u, v, w, 1.0f);
    }
    else if (state[my_offset] == 1 && count_live >= 6) {

        state_next[my_offset] = 0;
        pos[my_offset] = make_float4(1, 1, 1, 1.0f);
    }
    else {
        //printf("\nelse offset in cuda: %d, xyz %d %d %d, uvw %f %f %f", my_offset, x, y, z, u, v, w);
        state_next[my_offset] = state[my_offset];
        pos[my_offset] = make_float4(1, 1, 1, 1.0f);
    }

}


void launch_kernel(float4* pos, unsigned int mesh_width,
    unsigned int mesh_height, float time, int* d_state, int* d_state_next)
{
    // execute the kernel 
    dim3 block(8, 8, 8); 
    dim3 grid(mesh_width / block.x, mesh_width / block.y, mesh_height / block.z);
    simple_vbo_kernel << < grid, block >> > (pos, mesh_width, mesh_width, mesh_height, time, d_state, d_state_next);
}

bool checkHW(char* name, const char* gpuType, int dev)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    strcpy(name, deviceProp.name);

    if (!STRNCASECMP(deviceProp.name, gpuType, strlen(gpuType)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    char* ref_file = NULL;

    pArgc = &argc;
    pArgv = argv;

#if defined(__linux__)
    setenv("DISPLAY", ":0", 0);
#endif

    printf("%s starting...@main1 !!!\n", sSDKsample);

    if (argc > 1)
    {
        if (checkCmdLineFlag(argc, (const char**)argv, "file"))
        {
            // In this mode, we are running non-OpenGL and doing a compare of the VBO was generated correctly
            getCmdLineArgumentString(argc, (const char**)argv, "file", (char**)&ref_file);
        }
    }

    printf("\n");

    runTest(argc, argv, ref_file);

    printf("%s completed, returned %s\n", sSDKsample, (g_TotalErrors == 0) ? "OK" : "ERROR!");
    exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.f);

        sdkResetTimer(&timer);
    }

    char fps[256];
    sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz)", avgFPS);
    glutSetWindowTitle(fps);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int* argc, char** argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Cuda GL Interop (VBO)");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

    // initialize necessary OpenGL extensions
    if (!isGLVersionSupported(2, 0))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.1, 10.0);

    SDK_CHECK_ERROR_GL();

    return true;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
bool runTest(int argc, char** argv, char* ref_file)
{
    // Create the CUTIL timer
    sdkCreateTimer(&timer);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char**)argv);

    // command line mode only
    if (ref_file != NULL)
    {
        // create VBO
        checkCudaErrors(cudaMalloc((void**)&d_vbo_buffer, mesh_width * mesh_height * 4 * sizeof(float)));

        // run the cuda part
        runAutoTest(devID, argv, ref_file);

        // check result of Cuda step
        checkResultCuda(argc, argv, vbo);

        cudaFree(d_vbo_buffer);
        d_vbo_buffer = NULL;
    }
    else
    {
        // First initialize OpenGL context, so we can properly set the GL for CUDA.
        // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
        if (false == initGL(&argc, argv))
        {
            return false;
        }

        // register callbacks
        glutDisplayFunc(display);  
        glutKeyboardFunc(keyboard);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);

        glutCloseFunc(cleanup);

        // create VBO
        createVBO(&loc_vbo, &cuda_loc_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
  
        //allocate state grid on host
        state = (int*)malloc(mesh_width * mesh_width * mesh_height * sizeof(int));
        state_next = (int*)malloc(mesh_width * mesh_width * mesh_height * sizeof(int));

        //initialize first state
        initializeStartState(NULL, NULL, NULL, NULL);


        // run the cuda part
        runCuda(&cuda_loc_vbo_resource, state, state_next, d_state, d_state_next);
        
        // start rendering mainloop
        glutMainLoop();
    }

    return true;
}


////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource** vbo_resource, int* state, int* state_next, int* d_state, int* d_state_next)
{
    // map OpenGL buffer object for writing from CUDA
    float4* dptr;
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes,
        *vbo_resource));
    //printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

    //allocate device memory
    checkCudaErrors(cudaMalloc((void**)&d_state, mesh_width * mesh_width * mesh_height * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_state_next, mesh_width * mesh_width * mesh_height * sizeof(int)));
    //printf(state);printf(state_next);printf(d_state);printf(d_state_next);

    //fill device memory
    checkCudaErrors(cudaMemcpy(d_state, state, mesh_width * mesh_width * mesh_height * sizeof(int), cudaMemcpyHostToDevice));
    //printf("\nmemcpy happened fine\n");

    /* CUDA KERNEL CALL */
    int thread1D = 8; //MAX 64 for this GTX 1660Ti
    dim3 block(thread1D, thread1D, thread1D);
    dim3 grid(mesh_width / block.x, mesh_width / block.y, mesh_height / block.z);
    simple_vbo_kernel <<< grid, block >>> (dptr, mesh_width, mesh_width, mesh_height, g_fAnim, d_state, d_state_next);

    //next input state updated 
    checkCudaErrors(cudaMemcpy(state, d_state_next, mesh_width * mesh_width * mesh_height * sizeof(int), cudaMemcpyDeviceToHost));

    //int mid = mesh_width/2, x = mid, y = mid, z = mid, loc = (mesh_width * (y * mesh_width + x)) + z;
    //state[loc] = 1;
    //printf("\nvalue at state[mid] on host after cudaCpy = %d", state[loc]);

    //flush device memory
    checkCudaErrors(cudaFree(d_state));checkCudaErrors(cudaFree(d_state_next));

    //state_next freeing causing exceptions, TODO investigate reason
    //free(state_next); 

    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

#ifdef _WIN32
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) fopen_s(&fHandle, filename, mode)
#endif
#else
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) (fHandle = fopen(filename, mode))
#endif
#endif

void sdkDumpBin2(void* data, unsigned int bytes, const char* filename)
{
    printf("sdkDumpBin: <%s>\n", filename);
    FILE* fp;
    FOPEN(fp, filename, "wb");
    fwrite(data, bytes, 1, fp);
    fflush(fp);
    fclose(fp);
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runAutoTest(int devID, char** argv, char* ref_file)
{
    char* reference_file = NULL;
    void* imageData = malloc(mesh_width * mesh_height * sizeof(float));

    // execute the kernel
    //launch_kernel((float4*)d_vbo_buffer, mesh_width, mesh_height, g_fAnim);

    cudaDeviceSynchronize();
    getLastCudaError("launch_kernel failed");

    checkCudaErrors(cudaMemcpy(imageData, d_vbo_buffer, mesh_width * mesh_height * sizeof(float), cudaMemcpyDeviceToHost));

    sdkDumpBin2(imageData, mesh_width * mesh_height * sizeof(float), "simpleGL.bin");
    reference_file = sdkFindFilePath(ref_file, argv[0]);

    if (reference_file &&
        !sdkCompareBin2BinFloat("simpleGL.bin", reference_file,
            mesh_width * mesh_height * sizeof(float),
            MAX_EPSILON_ERROR, THRESHOLD, pArgv[0]))
    {
        g_TotalErrors++;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint* vbo, struct cudaGraphicsResource** vbo_res,
    unsigned int vbo_res_flags)
{
    assert(vbo);

    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    unsigned int size = mesh_width * mesh_width * mesh_height * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

    SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_res)
{

    // unregister this buffer object with CUDA
    checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    sdkStartTimer(&timer);

    // run CUDA kernel to generate vertex positions
    runCuda(&cuda_loc_vbo_resource, state, state_next, d_state, d_state_next);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, loc_vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(0.0, 1.0, 0.4);
    glPointSize(1.0f);
    glDrawArrays(GL_POINTS, 0, mesh_width * mesh_width * mesh_height); //added third multiplier 
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();

    g_fAnim += 0.01f;

    sdkStopTimer(&timer);
    computeFPS();
}

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    if (vbo)
    {
        deleteVBO(&vbo, cuda_vbo_resource);
        
    }
    if (loc_vbo) {
        deleteVBO(&loc_vbo, cuda_loc_vbo_resource);
    }
    free(state);free(state_next);
}


////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
    case (27):

        glutDestroyWindow(glutGetWindow());
        return;

    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1 << button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
    else if (mouse_buttons & 4)
    {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

////////////////////////////////////////////////////////////////////////////////
//! Check if the result is correct or write data to file for external
//! regression testing
////////////////////////////////////////////////////////////////////////////////
void checkResultCuda(int argc, char** argv, const GLuint& vbo)
{
    if (!d_vbo_buffer)
    {
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));

        // map buffer object
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        float* data = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);

        // check result
        if (checkCmdLineFlag(argc, (const char**)argv, "regression"))
        {
            // write file for regression test
            sdkWriteFile<float>("./data/regression.dat",
                data, mesh_width * mesh_height * 3, 0.0, false);
        }

        // unmap GL buffer object
        if (!glUnmapBuffer(GL_ARRAY_BUFFER))
        {
            fprintf(stderr, "Unmap buffer failed.\n");
            fflush(stderr);
        }

        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo,
            cudaGraphicsMapFlagsWriteDiscard));

        SDK_CHECK_ERROR_GL();
    }
}
