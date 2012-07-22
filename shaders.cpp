/*
 * GPU Smoldyn: Smoldyn algorithm ported to the GPU using CUDA 2.2
 * Writtern By Lorenzo Dematté, 2010-2011
 *
 * This file is part of GPU Smoldyn
 * 
 *     GPU Smoldyn is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 * 
 *     GPU Smoldyn is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with Foobar.  If not, see <http://www.gnu.org/licenses/>. 
 * 
 * Based on algorithm and source code of Smoldyn, written by Steve Andrews, 2003.
 * 
 * Portions taken by code examples in NVIDIA Whitepapers, GPU Gems 2 and 3, 
 * Copyright 1993-2009 NVIDIA Corporation, Addison-Wesley and the original authors. 
 * 
 */

#define STRINGIFY(A) #A

// vertex shader
const char *vertexShader = STRINGIFY(

#version 130\n

uniform float pointRadius;  // point size in world space
uniform float pointScale;   // scale to calculate size in pixels
uniform float densityScale;
uniform float densityOffset;

in int type;
flat out int particleType;


void main(void)
{
    // calculate window-space point size
    vec3 posEye = vec3(gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0));
    float dist = length(posEye);
    gl_PointSize = pointRadius * (pointScale / dist);

    //gl_TexCoord[0] = gl_MultiTexCoord0;
    gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);

    particleType = type;
    gl_FrontColor = vec4(1, 1, 1, 1);
}
);

// pixel shader for rendering points as shaded spheres
const char *spherePixelShader = STRINGIFY(

#version 130\n

uniform vec4 colors[25];

flat in int particleType;

void main()
{
    const vec3 lightDir = vec3(0.577, 0.577, 0.577);

    // calculate normal from texture coordinates
    vec3 N;
    N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(N.xy, N.xy);
    if (mag > 1.0) discard;   // kill pixels outside circle
    N.z = sqrt(1.0-mag);

    // calculate lighting
    float diffuse = max(0.0, dot(lightDir, N));

    //gl_FragColor = gl_Color * diffuse;
    //if (particleType < 0) discard;   // kill non-live types
    int index = particleType % 25;


    //gl_FragColor = vec4(0, 0, particleType, 1) * diffuse;
    //if (particleType == 16777216)
    //  gl_FragColor = vec4(0, 0, 1, 1) * diffuse;  
    //else //if (index == 2)
    //  gl_FragColor = vec4(1, 0, 0, 1) * diffuse;
    //else
    gl_FragColor = colors[index] * diffuse;
}
);
