// ============================================================================
//  FILENAME   : Render.cpp
//  PROJECT    : GPU Rendering Playground
//  AUTHOR     : Luc <lucdemercey@gmail.com>
//  CREATED    : 2025-10-16
//  UPDATED    : 2025-10-17
//  DESCRIPTION: Step 6 OpenGL - Lighting
// ============================================================================

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "Render.hpp"


glm::vec3 spiralCam(float time) {
	float radius = 20.0f;
	float heightAmplitude = 15.0f;
	float speedAngle = 0.3f;	// vitesse de rotation
	float speedHeight = 0.2f;	// vitesse de montee

	float angle = time * speedAngle;
	float y = sin(time * speedHeight) * heightAmplitude;
	float x = cos(angle) * radius;
	float z = sin(angle) * radius;

	return glm::vec3(x, y, z);
}

glm::vec3 computeCameraPos(float time) {
	float radius = 15.0f;
	float camX = sin(time) * radius;
	float camZ = cos(time) * radius;
	return glm::vec3(camX, 5.0f, camZ); // Y fixe pour garder la hauteur
}

glm::vec3 computeLightPos(float time) {
    float radius = 10.0f;
    float x = sin(time) * radius;
    float z = cos(time) * radius;
    return glm::vec3(x, 5.0f, z); // Y fixe ou variable
}


/* Camera mobile */

/*
void renderLoop(GLFWwindow* window, const std::vector<Shape> &shapes, unsigned int shaderProgram) {
	glUseProgram(shaderProgram);

	// Coordonnées de la lumière
	glm::vec3 lightPos(10.0f, 10.0f, 10.0f);

	// Projection et view matrices
	glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
		// Position de la camera dans le monde, target, le vecteur haut de la camera (ici y pointe vers le haut)
		// POur deplacer la camera : cameraPos, tourner la scene : cameraPos ou target

	

	while (!glfwWindowShouldClose(window)) {
		// Clear screen
		glClearColor(0.1f, 0.2f, 0.2f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST); // active le depth_test pour le rendu 3D

		// Calcul la position de la camera
		float time = glfwGetTime();
		glm::vec3 viewPos = computeCameraPos(time);
		glm::mat4 view = glm::lookAt(viewPos, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

		// Envoyer les uniforms globaux
		glUniform3f(glGetUniformLocation(shaderProgram, "lightPos"), lightPos.x, lightPos.y, lightPos.z);
		glUniform3f(glGetUniformLocation(shaderProgram, "viewPos"), viewPos.x, viewPos.y, viewPos.z);
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
		glUniform3f(glGetUniformLocation(shaderProgram, "lightColor"), 1.0f, 1.0f, 1.0f);

		// Dessin des shapes
		for (const auto &shape : shapes) {
			glBindVertexArray(shape.VAO);

			// Matrice model : faire tourner l’objet
			glm::mat4 model = glm::mat4(1.0f);
			if (shape._name == "sphere") {
				model = glm::translate(model, glm::vec3(5.0f, 0.0f, 0.0f));
			}
			
			// Envoyer les matrices au shader
			glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
				// Couleur de l’objet
			glUniform3f(glGetUniformLocation(shaderProgram, "objectColor"), 1.0f, 0.0f, 0.0f);

			// Dessiner
			if (shape.indexCount > 0)
				glDrawElements(GL_TRIANGLES, shape.indexCount, GL_UNSIGNED_INT, 0);
			else
				glDrawArrays(GL_TRIANGLES, 0, shape.vertexCount);
		}
		
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
}
*/

/* Camera fixe */

/*
void renderLoop(GLFWwindow* window, const std::vector<Shape> &shapes, unsigned int shaderProgram) {
	glUseProgram(shaderProgram);

	// Position fixe de la lumière et de la caméra
	glm::vec3 lightPos(10.0f, 10.0f, 10.0f);
	glm::vec3 viewPos(0.0f, 0.0f, 15.0f); // caméra fixe

	// Projection et view matrices
	glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
	glm::mat4 view = glm::lookAt(viewPos, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

	while (!glfwWindowShouldClose(window)) {
		// Clear screen
		glClearColor(0.1f, 0.2f, 0.2f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST); // active le depth_test pour le rendu 3D

		// Envoyer les uniforms globaux
		glUniform3f(glGetUniformLocation(shaderProgram, "lightPos"), lightPos.x, lightPos.y, lightPos.z);
		glUniform3f(glGetUniformLocation(shaderProgram, "viewPos"), viewPos.x, viewPos.y, viewPos.z);
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
		glUniform3f(glGetUniformLocation(shaderProgram, "lightColor"), 1.0f, 1.0f, 1.0f);

		for (const auto &shape : shapes) {
			glBindVertexArray(shape.VAO);

			// Matrice model : rotation du cube
			glm::mat4 model = glm::mat4(1.0f);
			float angleY = (float)glfwGetTime();
			if (shape._name == "sphere") {
				model = glm::translate(model, glm::vec3(5.0f, 0.0f, 0.0f));
				model = glm::rotate(model, angleY, glm::vec3(0, 1, 0));
			}
			if (shape._name == "cube") {
				// float angleX = (float)glfwGetTime() * 0.2f;
				model = glm::rotate(model, angleY, glm::vec3(0, 1, 0));
				// model = glm::rotate(model, angleX, glm::vec3(1, 0, 0));
			}
			glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));

			// Couleur de l’objet
			glUniform3f(glGetUniformLocation(shaderProgram, "objectColor"), 1.0f, 0.0f, 0.0f);

			// Dessiner
			if (shape.indexCount > 0)
				glDrawElements(GL_TRIANGLES, shape.indexCount, GL_UNSIGNED_INT, 0);
			else
				glDrawArrays(GL_TRIANGLES, 0, shape.vertexCount);
		}

		glfwSwapBuffers(window);
		glfwPollEvents();
	}
}
*/

/* Lumiere mobile */

/*
void renderLoop(GLFWwindow* window, const std::vector<Shape> &shapes, unsigned int shaderProgram) {
	glUseProgram(shaderProgram);

	// Coordonnées de la camera
	glm::vec3 viewPos(0.0f, 3.0f, 12.0f);

	// Projection et view matrices
	glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
		// Position de la camera dans le monde, target, le vecteur haut de la camera (ici y pointe vers le haut)
		// POur deplacer la camera : cameraPos, tourner la scene : cameraPos ou target
	glm::mat4 view = glm::lookAt(viewPos, glm::vec3(0.0f), glm::vec3(0.0f,1.0f,0.0f));
	glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));

	while (!glfwWindowShouldClose(window)) {
		// Clear screen
		glClearColor(0.1f, 0.2f, 0.2f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST); // active le depth_test pour le rendu 3D

		// Calcul la position de la lumiere
		float time = glfwGetTime();
		glm::vec3 lightPos = computeLightPos(time);

		// Envoyer les uniforms globaux
		glUniform3f(glGetUniformLocation(shaderProgram, "lightPos"), lightPos.x, lightPos.y, lightPos.z);
		glUniform3f(glGetUniformLocation(shaderProgram, "viewPos"), viewPos.x, viewPos.y, viewPos.z);
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
		glUniform3f(glGetUniformLocation(shaderProgram, "lightColor"), 1.0f, 1.0f, 1.0f);

		// Dessin des shapes
		for (const auto &shape : shapes) {
			glBindVertexArray(shape.VAO);

			// Matrice model : faire tourner l’objet
			glm::mat4 model = glm::mat4(1.0f);
			if (shape._name == "sphere") {
				model = glm::translate(model, glm::vec3(5.0f, 0.0f, 0.0f));
			}
			if (shape._name == "cube") {
				model = glm::translate(model, glm::vec3(-5.0f, 0.0f, 0.0f));
			}
			
			// Envoyer les matrices au shader
			glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
				// Couleur de l’objet
			glUniform3f(glGetUniformLocation(shaderProgram, "objectColor"), 1.0f, 0.0f, 0.0f);

			// Dessiner
			if (shape.indexCount > 0)
				glDrawElements(GL_TRIANGLES, shape.indexCount, GL_UNSIGNED_INT, 0);
			else
				glDrawArrays(GL_TRIANGLES, 0, shape.vertexCount);
		}
		
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
}
*/

/* Lumiere changeante */
/*
void renderLoop(GLFWwindow* window, const std::vector<Shape> &shapes, unsigned int shaderProgram) {
	glUseProgram(shaderProgram);

	// Position fixe de la lumière et de la caméra
	glm::vec3 lightPos(10.0f, 10.0f, 10.0f);
	glm::vec3 viewPos(0.0f, 0.0f, 15.0f); // caméra fixe

	// Projection et view matrices
	glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
	glm::mat4 view = glm::lookAt(viewPos, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

	while (!glfwWindowShouldClose(window)) {
		// Clear screen
		glClearColor(0.1f, 0.2f, 0.2f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST); // active le depth_test pour le rendu 3D

		float time = glfwGetTime();

		// Couleur de lumière dynamique : variation harmonique
		float r = (sin(time * 0.8f) + 1.0f) * 0.5f; // varie de 0 à 1
		float g = (sin(time * 0.6f + 2.0f) + 1.0f) * 0.5f;
		float b = (sin(time * 1.0f + 4.0f) + 1.0f) * 0.5f;

		// Format jour & nuit
		float t = fmod(glfwGetTime(), 10.0f) / 10.0f; // cycle de 10 secondes
		glm::vec3 lightColor = glm::mix(glm::vec3(1.0f, 0.5f, 0.2f), glm::vec3(0.2f, 0.4f, 1.0f), abs(sin(t * glm::pi<float>())));


		// Envoyer les uniforms globaux
		glUniform3f(glGetUniformLocation(shaderProgram, "lightPos"), lightPos.x, lightPos.y, lightPos.z);
		glUniform3f(glGetUniformLocation(shaderProgram, "viewPos"), viewPos.x, viewPos.y, viewPos.z);
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
		// glUniform3f(glGetUniformLocation(shaderProgram, "lightColor"), r, g, b);
		glUniform3f(glGetUniformLocation(shaderProgram, "lightColor"), lightColor.r, lightColor.g, lightColor.b);


		for (const auto &shape : shapes) {
			glBindVertexArray(shape.VAO);

			// Matrice model : rotation du cube
			glm::mat4 model = glm::mat4(1.0f);
			float angleY = (float)glfwGetTime();
			if (shape._name == "sphere") {
				model = glm::translate(model, glm::vec3(5.0f, 0.0f, 0.0f));
				// model = glm::rotate(model, angleY, glm::vec3(0, 1, 0));
			}
			if (shape._name == "cube") {
				float angleX = (float)glfwGetTime() * 0.2f;
				model = glm::rotate(model, angleY, glm::vec3(0, 1, 0));
				model = glm::rotate(model, angleX, glm::vec3(1, 0, 0));
			}
			glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));

			// Couleur de l’objet
			glUniform3f(glGetUniformLocation(shaderProgram, "objectColor"), 1.0f, 0.0f, 0.0f);

			// Dessiner
			if (shape.indexCount > 0)
				glDrawElements(GL_TRIANGLES, shape.indexCount, GL_UNSIGNED_INT, 0);
			else
				glDrawArrays(GL_TRIANGLES, 0, shape.vertexCount);
		}

		glfwSwapBuffers(window);
		glfwPollEvents();
	}
}
*/

/* Plusieurs lumieres */
// Attention, on change le fragment_shader !!
/*
void renderLoop(GLFWwindow* window, const std::vector<Shape> &shapes, unsigned int shaderProgram) {
	glUseProgram(shaderProgram);

	// Position fixe de la lumière et de la caméra
	glm::vec3 viewPos(0.0f, 0.0f, 15.0f); 		// caméra fixe
	std::vector<glm::vec3> lightPositions = { 	// Multiple light
		glm::vec3(10.0f, 10.0f, 10.0f),
		glm::vec3(0.0f, -5.0f, 5.0f),
		glm::vec3(-5.0f, 0.0f, 0.0f)
	};

	std::vector<glm::vec3> lightColors = {
		glm::vec3(0.6f, 0.9f, 0.4f),
		glm::vec3(0.0f, 1.0f, 1.0f),
		glm::vec3(0.7f, 0.5f, 0.3f)
	};



	// Projection et view matrices
	glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
	glm::mat4 view = glm::lookAt(viewPos, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

	while (!glfwWindowShouldClose(window)) {
		// Clear screen
		glClearColor(0.1f, 0.2f, 0.2f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST); // active le depth_test pour le rendu 3D

		// Envoyer les uniforms globaux
		glUniform3f(glGetUniformLocation(shaderProgram, "viewPos"), viewPos.x, viewPos.y, viewPos.z);
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
		
		glUniform1i(glGetUniformLocation(shaderProgram, "numLights"), lightPositions.size());
		for (int i = 0; i < lightPositions.size(); ++i) {
			std::string posName = "lights[" + std::to_string(i) + "].position";
			std::string colorName = "lights[" + std::to_string(i) + "].color";

			glUniform3fv(glGetUniformLocation(shaderProgram, posName.c_str()), 1, glm::value_ptr(lightPositions[i]));
			glUniform3fv(glGetUniformLocation(shaderProgram, colorName.c_str()), 1, glm::value_ptr(lightColors[i]));
		}


		for (const auto &shape : shapes) {
			glBindVertexArray(shape.VAO);

			// Matrice model : rotation du cube
			glm::mat4 model = glm::mat4(1.0f);
			float angleY = (float)glfwGetTime();
			if (shape._name == "sphere") {
				model = glm::translate(model, glm::vec3(5.0f, 0.0f, 0.0f));
				// model = glm::rotate(model, angleY, glm::vec3(0, 1, 0));
			}
			if (shape._name == "cube") {
				// float angleX = (float)glfwGetTime() * 0.2f;
				model = glm::rotate(model, angleY, glm::vec3(0, 1, 0));
				// model = glm::rotate(model, angleX, glm::vec3(1, 0, 0));
			}
			glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));

			// Couleur de l’objet
			glUniform3f(glGetUniformLocation(shaderProgram, "objectColor"), 1.0f, 0.0f, 0.0f);

			// Dessiner
			if (shape.indexCount > 0)
				glDrawElements(GL_TRIANGLES, shape.indexCount, GL_UNSIGNED_INT, 0);
			else
				glDrawArrays(GL_TRIANGLES, 0, shape.vertexCount);
		}

		glfwSwapBuffers(window);
		glfwPollEvents();
	}
}
*/

/* Spotlight */
// Attention, on chane le fragment shader !!
/*
void renderLoop(GLFWwindow* window, const std::vector<Shape> &shapes, unsigned int shaderProgram) {
	glUseProgram(shaderProgram);

	// Calcul du spotlight
	glm::vec3 spotPos = glm::vec3(10.0f, 10.0f, 10.0f);
	glm::vec3 spotDir = glm::vec3(-1.0f, -1.0f, -1.0f); // direction vers l'origine
	float cutOff = glm::cos(glm::radians(12.5f));
	float outerCutOff = glm::cos(glm::radians(17.5f));

	// Projection et view matrices
	glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
		// Position de la camera dans le monde, target, le vecteur haut de la camera (ici y pointe vers le haut)
		// POur deplacer la camera : cameraPos, tourner la scene : cameraPos ou target

	

	while (!glfwWindowShouldClose(window)) {
		// Clear screen
		glClearColor(0.1f, 0.2f, 0.2f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST); // active le depth_test pour le rendu 3D

		// Calcul la position de la camera
		float time = glfwGetTime();
		glm::vec3 viewPos = computeCameraPos(time);
		glm::mat4 view = glm::lookAt(viewPos, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

		
		// Envoyer les uniforms globaux
		glUniform3f(glGetUniformLocation(shaderProgram, "viewPos"), viewPos.x, viewPos.y, viewPos.z);
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
		glUniform3f(glGetUniformLocation(shaderProgram, "lightColor"), 1.0f, 1.0f, 1.0f);

		glUniform3fv(glGetUniformLocation(shaderProgram, "spotlight.position"), 1, glm::value_ptr(spotPos));
		glUniform3fv(glGetUniformLocation(shaderProgram, "spotlight.direction"), 1, glm::value_ptr(spotDir));
		glUniform3f(glGetUniformLocation(shaderProgram, "spotlight.color"), 1.0f, 1.0f, 1.0f);
		glUniform1f(glGetUniformLocation(shaderProgram, "spotlight.cutOff"), cutOff);
		glUniform1f(glGetUniformLocation(shaderProgram, "spotlight.outerCutOff"), outerCutOff);


		// Dessin des shapes
		for (const auto &shape : shapes) {
			glBindVertexArray(shape.VAO);

			// Matrice model : faire tourner l’objet
			glm::mat4 model = glm::mat4(1.0f);
			if (shape._name == "sphere") {
				model = glm::translate(model, glm::vec3(5.0f, 0.0f, 0.0f));
			}
			
			// Envoyer les matrices au shader
			glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
			glUniform3f(glGetUniformLocation(shaderProgram, "objectColor"), 1.0f, 0.0f, 0.0f);

			// Dessiner
			if (shape.indexCount > 0)
				glDrawElements(GL_TRIANGLES, shape.indexCount, GL_UNSIGNED_INT, 0);
			else
				glDrawArrays(GL_TRIANGLES, 0, shape.vertexCount);
		}
		
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
}
*/

/* Animation spirale */

void renderLoop(GLFWwindow* window, const std::vector<Shape> &shapes, unsigned int shaderProgram) {
	glUseProgram(shaderProgram);

	// Coordonnées de la lumière
	glm::vec3 lightPos(10.0f, 10.0f, 10.0f);

	// Projection et view matrices
	glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
		// Position de la camera dans le monde, target, le vecteur haut de la camera (ici y pointe vers le haut)
		// POur deplacer la camera : cameraPos, tourner la scene : cameraPos ou target

	

	while (!glfwWindowShouldClose(window)) {
		// Clear screen
		glClearColor(0.1f, 0.2f, 0.2f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST); // active le depth_test pour le rendu 3D

		// Calcul la position de la camera
		float time = glfwGetTime();
		glm::vec3 viewPos = spiralCam(time);
		glm::mat4 view = glm::lookAt(viewPos, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

		// Balayage de toute la piece
		// glm::vec3 viewPos(0.0f, 0.0f, 4.0f); // caméra au centre
		// float radius = 10.0f; // distance du point regardé

		// float theta = time * 0.5f; // vitesse de rotation
		// float phi   = sin(time * 0.2f) * glm::radians(30.0f); // variation verticale

		// glm::vec3 target(
		// 	cos(theta) * radius,
		// 	viewPos.y + sin(phi),
		// 	sin(theta) * radius
		// );
		// glm::mat4 view = glm::lookAt(viewPos, target, glm::vec3(0, 1, 0));

		// Envoyer les uniforms globaux
		glUniform3f(glGetUniformLocation(shaderProgram, "lightPos"), lightPos.x, lightPos.y, lightPos.z);
		glUniform3f(glGetUniformLocation(shaderProgram, "viewPos"), viewPos.x, viewPos.y, viewPos.z);
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
		glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
		glUniform3f(glGetUniformLocation(shaderProgram, "lightColor"), 1.0f, 1.0f, 1.0f);

		// Dessin des shapes
		for (const auto &shape : shapes) {
			glBindVertexArray(shape.VAO);

			// Matrice model : faire tourner l’objet
			glm::mat4 model = glm::mat4(1.0f);
			if (shape._name == "sphere") {
				model = glm::translate(model, glm::vec3(5.0f, 0.0f, 0.0f));
			}
			if (shape._name == "cube") {
				float angleY = (float)glfwGetTime() * 0.5f;
				float angleX = (float)glfwGetTime() * 0.3f;
				model = glm::rotate(model, angleY, glm::vec3(0, 1, 0));
				model = glm::rotate(model, angleX, glm::vec3(1, 0, 0));
			}
			if (shape._name == "cube2") {
				float angle = (float)glfwGetTime() * 0.5f;
				model = glm::translate(model, glm::vec3(0.0f, 0.0f, 15.0f));
				model = glm::rotate(model, angle, glm::vec3(0, 1, 1));
			}
			if (shape._name == "orbite") {
				float radius = 5.0f;
				float angle = time * 0.5f;
				// Translation pour mettre l’objet sur le cercle
				model = glm::rotate(model, angle, glm::vec3(1,0,0));
				model = glm::translate(model, glm::vec3(0.0f, cos(angle) * radius, sin(angle) * radius));
			}
			
			// Envoyer les matrices au shader
			glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
				// Couleur de l’objet
			glUniform3f(glGetUniformLocation(shaderProgram, "objectColor"), 1.0f, 0.0f, 0.0f);

			// Dessiner
			if (shape.indexCount > 0)
				glDrawElements(GL_TRIANGLES, shape.indexCount, GL_UNSIGNED_INT, 0);
			else
				glDrawArrays(GL_TRIANGLES, 0, shape.vertexCount);
		}
		
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
}

