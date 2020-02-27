import numpy as np

def rotation_xyz(a=np.zeros(3),b=2*np.pi*np.ones(3)):
    #angles =  np.random.normal(np.pi, 3*np.pi, 3) 
    angles = np.zeros(3)
    for i in range(3):
        angles[i] =  np.random.uniform(a[i], b[i], 1) 
    
    cos_x, sin_x = np.cos(angles[0]), np.sin(angles[0])
    cos_y, sin_y = np.cos(angles[1]), np.sin(angles[1])
    cos_z, sin_z = np.cos(angles[2]), np.sin(angles[2])

    r_x = np.array([[1, 0, 0],
                    [0, cos_x, -sin_x],
                    [0, sin_x, cos_x]])

    r_y = np.array([[cos_y, 0, sin_y],
                    [0, 1, 0],
                    [-sin_y, 0, cos_y]])

    r_z = np.array([[cos_z, sin_z, 0],
                    [-sin_z, cos_z, 0],
                    [0, 0, 1]])

    rotation_matrix = np.dot(np.dot(r_z, r_y),r_x)
            

    return rotation_matrix

def rotation_axis_angle(a=np.zeros(3),b=2*np.pi*np.ones(3)):
    #angles =  np.random.normal(np.pi, 3*np.pi, 3) 
    angles = np.zeros(3)
    for i in range(3):
        angles[i] =  np.random.uniform(a[i], b[i], 1) 
    cos1, sin1 = np.cos(angles[0]), np.sin(angles[0])
    cos2, sin2 = np.cos(angles[1]), np.sin(angles[1])
    u = np.array([sin1, cos1*sin2, cos1*cos2]) # normalized vector as axis

    K = np.array([[0, -u[2],u[1]],
                  [u[2], 0, -u[0]],
                  [-u[1], u[0], 0]])
    theta = np.random.uniform(a[2],b[2],1) #rotate bt theta
    rotation_matrix = np.identity(3) + np.sin(theta)*K + (1 - np.cos(theta))* np.dot(K, K)

    return rotation_matrix


def rotation(a=np.zeros(3),b=2*np.pi*np.ones(3)): # equals to rotation_xyz
    angles = np.zeros(3)
    for i in range(3):
        angles[i] =  np.random.uniform(a[i], b[i], 1) 
    
    cos1, sin1 = np.cos(angles[0]), np.sin(angles[0])
    cos2, sin2 = np.cos(angles[1]), np.sin(angles[1])
    cos3, sin3 = np.cos(angles[2]), np.sin(angles[2])
    rotation_matrix = np.array([[cos1*cos3-cos2*sin1*sin3, -cos2*cos3*sin1-cos1*sin3,  sin1*sin2],
                                [cos3*sin1+cos1*cos2*sin3,  cos1*cos2*cos3-sin1*sin3, -cos1*sin2],
                                [      sin2*sin3         ,       cos3*sin2          ,    cos2   ]])
   

    return rotation_matrix

def reflection(a=np.zeros(3),b=2*np.pi*np.ones(3)):
    ## matrix = I - 2u^tu, reflection on a plane cross 0 with normal vector u
    angles = np.zeros(3)
    for i in range(3):
        angles[i] =  np.random.uniform(a[i], b[i], 1) 
    cos1, sin1 = np.cos(angles[0]), np.sin(angles[0])
    cos2, sin2 = np.cos(angles[1]), np.sin(angles[1])
    u = np.array([[sin1, cos1*sin2, cos1*cos2]]) # normalized vector as axis
    
    matrix = np.identity(3) - 2 * np.dot(u.transpose(),u)
    return matrix

def ref_rot(a=np.zeros(3),b=2*np.pi*np.ones(3)):
    ## matrix = I - 2u^tu
    angles = np.zeros(3)
    for i in range(3):
        angles[i] =  np.random.uniform(a[i], b[i], 1) 
    cos1, sin1 = np.cos(angles[0]), np.sin(angles[0])
    cos2, sin2 = np.cos(angles[1]), np.sin(angles[1])
    u = np.array([[sin1, cos1*sin2, cos1*cos2]]) # normalized vector as axis
    cos_z, sin_z = np.cos(angles[2]), np.sin(angles[2])
    r_z = np.array([[cos_z, sin_z, 0],
                    [-sin_z, cos_z, 0],
                    [0, 0, 1]])
    matrix = np.identity(3) - 2 * np.dot(u.transpose(),u)
    matrix = np.dot(r_z, matrix) #reflection then rotate by axis-z

    return matrix

# Cartan–Dieudonné theorem, establishes that every orthogonal transformation in an n-dimensional 
# symmetric bilinear space can be described as the composition of at most n reflections
# in three-dimensional Euclidean space, every orthogonal transformation can be described as a 
# single reflection, a rotation (2 reflections), or an improper rotation (3 reflections) 
# -- from https://en.wikipedia.org/wiki/Cartan%E2%80%93Dieudonn%C3%A9_theorem