#version 330 core

in vec3 FragPos;
in vec3 Normal;
in vec3 VertexColor;

out vec4 FragColor;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;

void main()
{
    // Normale normalisée
    vec3 norm = normalize(Normal);

    // Direction lumière
    vec3 lightDir = normalize(lightPos - FragPos);

    // Composante diffuse
    float diff = max(dot(norm, lightDir), 0.0);

    // Composante ambient
    vec3 ambient = 0.1 * lightColor;

    // Composante diffuse finale
    vec3 diffuse = diff * lightColor;

    // Composante speculaire
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = spec * lightColor;

    vec3 result = (ambient + diffuse + specular) * VertexColor;
    FragColor = vec4(result, 1.0);
}
