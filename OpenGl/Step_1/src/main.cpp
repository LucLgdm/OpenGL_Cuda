// ============================================================================
//  FILENAME   : main.cpp
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-09-11
//  UPDATED    : 2025-09-11
//  DESCRIPTION: Step 1 OpenGL - Hello Triangle
// ============================================================================

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <iostream>
using namespace std;

// -------------------------------------------------------------
// Callbacks
// -------------------------------------------------------------
/**********************************************************
 * Set the viewport to cover the entire window
 **********************************************************/
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

// -------------------------------------------------------------
// Initialization
// -------------------------------------------------------------
GLFWwindow* initWindow(int width, int height, const char* title)
{
    if (!glfwInit()) return nullptr;
	/**********************************************************
	 * Configure the window and OpenGL context:
	 *  - OpenGL version 3.3
	 *  - Use the Core Profile
	 * Must be done before glfwCreateWindow()
	 **********************************************************/
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window) { glfwTerminate(); return nullptr; }

	/***********************************************************
	 * Make this OpenGL context current for all OpenGL commands
	 ***********************************************************/
    glfwMakeContextCurrent(window);

	/***********************************************************
	 * Automatically handle window resizing
	 ***********************************************************/
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n";
        return nullptr;
    }
    return window;
}

// -------------------------------------------------------------
// Shaders
// -------------------------------------------------------------
/*********************************************************************
 * A shader is a small program that runs directly
 * on the GPU to process rendering data.
 * 
 * Provides full control over how each vertex & fragment is processed.
 * Enables complex visual effects, lighting, textures, simulations.
 * Makes the GPU pipeline fully programmable,
 * unlike the old fixed-function pipeline.
 *********************************************************************/

unsigned int createShaderProgram()
{
	/**********************************************************
	 * Vertex shader
	 * Runs once per vertex.
	 * Determines the final position of a vertex on the screen.
	 **********************************************************/
    const char* vertexShaderSource = R"(
        #version 330 core
        layout(location = 0) in vec2 aPos;
        void main() {
            gl_Position = vec4(aPos, 0.0, 1.0);
        }
    )";
	/**********************************************************
	 * Fragment shader
	 * Runs once per fragment (potential pixel).
	 * Determines the final color of the fragment.
	 **********************************************************/
    const char* fragmentShaderSource = R"(
        #version 330 core
        out vec4 FragColor;
        void main() {
            FragColor = vec4(1.0, 0.5, 0.2, 1.0);
        }
    )";

	/**********************************************************
	 * Writes the shader code and compiles it
	 **********************************************************/
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

	/**********************************************************
	 * Creates a shader program, attach shaders and link them
	 **********************************************************/
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

// -------------------------------------------------------------
// Vertex Array Object & Vertex Buffer Object
// -------------------------------------------------------------
/***********************************************************************
 * A VBO is a GPU memory buffer that stores vertex data: 
 * positions, colors, texture coordinates, normals, etc.
 *  - It lives on the GPU, so drawing is very fast.
 *  - You can think of it like an array of vertices
 * 		that the GPU can read directly.
 *  - You can update it dynamically (for animations) or keep it static.
 * 
 * A VAO is an object that stores the configuration of your vertex inputs.
 * It remembers:
 * 	- Which VBO is bound.
 *  - How vertex attributes are laid out (position, color, etc.).
 *  - Any enabled attributes
 ***********************************************************************/
void createTriangle(unsigned int &VAO, unsigned int &VBO, float vertices[], size_t vertexCount)
{

	/****************************************************
	 * Gernerate and bind the VAO
	 * VAO stores the configuration of vertex attributes
	 ****************************************************/
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
	
	/****************************************************
	 * Generate and bind the VBO
     * VBO stores the actual vertex data in GPU memory
     * Define the vertex attributes:
	 * 	location, size, type, stride, offset
	 ****************************************************/
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, vertexCount * sizeof(float), vertices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	// Optional: unbind VBO and VAO to prevent accidental modification
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

// -------------------------------------------------------------
// Render loop
// -------------------------------------------------------------
void renderLoop(GLFWwindow* window, unsigned int VAO, unsigned int shaderProgram)
{
    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 9); // n vertices,  starting from index 0

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

// -------------------------------------------------------------
// Cleanup
// -------------------------------------------------------------
void cleanup(unsigned int VAO, unsigned int VBO, unsigned int shaderProgram, GLFWwindow* window)
{
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);
    glfwDestroyWindow(window);
    glfwTerminate();
}

// -------------------------------------------------------------
// Main
// -------------------------------------------------------------
int main()
{
    GLFWwindow* window = initWindow(800, 600, "Hello Triangle");
    if (!window) return -1;

    unsigned int shaderProgram = createShaderProgram();

	float s = 0.1f;
	float h = sqrt(3.0f) / 2.0f * s;

	float triangleVertices[] = {
		-0.4f, -0.5f,
		0.5f , -0.5f,
		0.5f ,  0.4f,
		-0.5f, -0.4f,
		0.4f ,  0.5f,
		-0.5f,  0.5f,
		0.0f,  2.0f/3.0f * h,
		-s/2.0f, -1.0f/3.0f * h,
		s/2.0f, -1.0f/3.0f * h};

    unsigned int VAO, VBO;
    createTriangle(VAO, VBO, triangleVertices, sizeof(triangleVertices));

    renderLoop(window, VAO, shaderProgram);

    cleanup(VAO, VBO, shaderProgram, window);
    return 0;
}
