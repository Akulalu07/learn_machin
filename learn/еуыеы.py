import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glBegin(GL_TRIANGLES)
        glVertex3f(-1, -1, -5)
        glVertex3f(1, -1, -5)
        glVertex3f(0, 1, -5)
        glEnd()
        pygame.display.flip()
        pygame.time.wait(10)

if __name__ == "__main__":
    main()
