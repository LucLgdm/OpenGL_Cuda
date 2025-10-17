#version 330 core

in vec3 FragPos;
in vec3 Normal;
in vec3 VertexColor;

out vec4 FragColor;

struct Light {
	vec3 position;
	vec3 color;
};

#define MAX_LIGHTS 4
uniform Light lights[MAX_LIGHTS];
uniform int numLights;

uniform vec3 viewPos;
uniform vec3 objectColor;

void main()
{
	vec3 norm = normalize(Normal);
	vec3 viewDir = normalize(viewPos - FragPos);

	vec3 result = vec3(0.0);

	for (int i = 0; i < numLights; ++i) {
		vec3 lightDir = normalize(lights[i].position - FragPos);

		// --- Composantes de l’éclairage ---
		float diff = max(dot(norm, lightDir), 0.0);
		vec3 reflectDir = reflect(-lightDir, norm);
		float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);

		vec3 ambient  = 0.1 * lights[i].color;
		vec3 diffuse  = diff * lights[i].color;
		vec3 specular = 0.5 * spec * lights[i].color;

		result += (ambient + diffuse + specular) * VertexColor;
	}

	FragColor = vec4(result, 1.0);

}
