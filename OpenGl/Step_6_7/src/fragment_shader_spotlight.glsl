#version 330 core

in vec3 FragPos;
in vec3 Normal;
in vec3 VertexColor;

out vec4 FragColor;

uniform vec3 viewPos;
uniform vec3 objectColor;

// Spotlight struct
struct Spotlight {
    vec3 position;
    vec3 direction;
    vec3 color;
    float cutOff;      // cos(angle intérieur)
    float outerCutOff; // cos(angle extérieur)
};

uniform Spotlight spotlight;

void main()
{
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(spotlight.position - FragPos);

    // Angle entre le spotlight et le fragment
    float theta = dot(lightDir, normalize(-spotlight.direction));
    float epsilon = spotlight.cutOff - spotlight.outerCutOff;
    float intensity = clamp((theta - spotlight.outerCutOff) / epsilon, 0.0, 1.0);

    // Calcul diffus et specular
    float diff = max(dot(norm, lightDir), 0.0);

    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);

    vec3 ambient  = 0.1 * spotlight.color * intensity;
    vec3 diffuse  = diff * spotlight.color * intensity;
    vec3 specular = 0.5 * spec * spotlight.color * intensity;

    vec3 result = (ambient + diffuse + specular) * VertexColor;

    FragColor = vec4(result, 1.0);
}
