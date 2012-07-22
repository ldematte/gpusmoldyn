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

#include <GL/glew.h>

#include <math.h>
#include <assert.h>
#include <stdio.h>

#include "render_particles.h"
#include "shaders.h"

#ifndef M_PI
#define M_PI    3.1415926535897932384626433832795
#endif

ParticleRenderer::ParticleRenderer()
: m_pos(0),
  m_numParticles(0),
  m_pointSize(1.0f),
  m_particleRadius(0.125f * 0.5f),
  m_program(0),
  m_vbo(0),
  m_typesVBO(0)
{
    _initGL();
}

ParticleRenderer::~ParticleRenderer()
{
    m_pos = 0;
}

void ParticleRenderer::setPositions(float *pos, int numParticles)
{
    m_pos = pos;
    m_numParticles = numParticles;
}

void ParticleRenderer::setBufferObjects(unsigned int posVBO, unsigned int typesVBO, int numParticles)
{
    m_vbo = posVBO;
    m_typesVBO = typesVBO;
    m_numParticles = numParticles;
}

void ParticleRenderer::_drawPoints()
{
    if (!m_vbo)
    {
        glBegin(GL_POINTS);
        {
            int k = 0;
            for (int i = 0; i < m_numParticles; ++i)
            {
                glVertex3fv(&m_pos[k]);
                k += 4;
            }
        }
        glEnd();
    }
    else
    {
        glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_vbo);
        glVertexPointer(4, GL_FLOAT, 0, 0);
        glEnableClientState(GL_VERTEX_ARRAY);   

        

        if (m_typesVBO) 
        {
            //glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_colorVBO);
            //glColorPointer(4, GL_FLOAT, 0, 0);            
            //glEnableClientState(GL_COLOR_ARRAY);

          

            glBindBuffer(GL_ARRAY_BUFFER, m_typesVBO);

            //void* mem = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY_ARB);
            //GLenum err = glGetError();
            //glUnmapBuffer(GL_ARRAY_BUFFER);
            
            //bind to TEX1, not TEX0!!
            //glClientActiveTextureARB(GL_TEXTURE1_ARB);
            //glTexCoordPointer(1, GL_INT, 0, 0);
            //glEnableClientState(GL_TEXTURE_COORD_ARRAY);

            int loc = glGetAttribLocation(m_program, "type");
            glEnableVertexAttribArray(loc);
            glVertexAttribPointer(loc, 1, GL_INT, GL_FALSE, 0, 0);
        }

        glDrawArrays(GL_POINTS, 0, m_numParticles);


        glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
        glDisableClientState(GL_VERTEX_ARRAY); 
        glDisableVertexAttribArray(glGetAttribLocation(m_program, "type"));
        //glDisableClientState(GL_COLOR_ARRAY); 
        //glDisableClientState(GL_TEXTURE_COORD_ARRAY);
        //glClientActiveTextureARB(GL_TEXTURE0_ARB); 
    }
}

void ParticleRenderer::display(DisplayMode mode /* = PARTICLE_POINTS */)
{
    switch (mode)
    {
    case PARTICLE_POINTS:
        glColor3f(1, 1, 1);
        glPointSize(m_pointSize);
        _drawPoints();
        break;

    default:
    case PARTICLE_SPHERES:
        glEnable(GL_POINT_SPRITE_ARB);
        glClientActiveTextureARB(GL_TEXTURE0_ARB); 
        glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
        glDepthMask(GL_TRUE);
        glEnable(GL_DEPTH_TEST);

        glUseProgram(m_program);
        glUniform1f( glGetUniformLocation(m_program, "pointScale"), m_window_h / tanf(m_fov*0.5f*(float)M_PI/180.0f) );
        glUniform1f( glGetUniformLocation(m_program, "pointRadius"), m_particleRadius );

       float colors[] = { 0, 1, 0, 1, 
                          1, 0, 0, 1,
                          0, 0, 1, 1,
                          1, 1, 0, 1,
                          0, 1, 1, 1,
                          1, 0, 1, 1,
                          0, 1, 0, 1, 
                          1, 0, 0, 1,
                          0, 0, 1, 1,
                          1, 1, 0, 1,
                          0, 1, 1, 1,
                          1, 0, 1, 1,
                          0, 1, 0, 1, 
                          1, 0, 0, 1,
                          0, 0, 1, 1,
                          1, 1, 0, 1,
                          0, 1, 1, 1,
                          1, 0, 1, 1,
                          0, 1, 0, 1, 
                          1, 0, 0, 1,
                          0, 0, 1, 1,
                          1, 1, 0, 1,
                          0, 1, 1, 1,
                          1, 0, 1, 1,
                          1, 1, 1, 0
       };

        glUniform4fv(glGetUniformLocation(m_program, "colors"), 25, colors);


        glColor3f(1, 1, 1);
        _drawPoints();

        glUseProgram(0);
        glDisable(GL_POINT_SPRITE_ARB);
        break;
    }
}

GLuint
ParticleRenderer::_compileProgram(const char *vsource, const char *fsource)
{
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vertexShader, 1, &vsource, 0);
    glShaderSource(fragmentShader, 1, &fsource, 0);
    
    glCompileShader(vertexShader);
    glCompileShader(fragmentShader);

    GLuint program = glCreateProgram();

    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);

    //glBindAttribLocation(program, 1, "type");

    glLinkProgram(program);

    // check if program linked
    GLint success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);

    if (!success) {
        char temp[1024];
        glGetProgramInfoLog(program, 1024, 0, temp);
        printf("Failed to link program:\n%s\n", temp);
        glDeleteProgram(program);
        program = 0;
    }

    return program;
}

void ParticleRenderer::_initGL()
{
    m_program = _compileProgram(vertexShader, spherePixelShader);



#if !defined(__APPLE__) && !defined(MACOSX)
    glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
    glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);
#endif
}
