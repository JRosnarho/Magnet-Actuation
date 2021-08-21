import numpy as np
# Import Cosserat Rod Class
from elastica.rod.cosserat_rod import CosseratRod
# Importing magpylib and matplotlib for magnet and plotting purposes
import matplotlib.pyplot as plt
from magpylib.source.magnet import Box, Cylinder, Sphere
from magpylib import Collection, displaySystem
from matplotlib.animation import FuncAnimation, PillowWriter


# Function to create a skew symmetric matrix
def skew(mat):
    skewed_mat = np.zeros((3, 3))
    skewed_mat[0, 1] = -mat[2]
    skewed_mat[0, 2] = mat[1]
    skewed_mat[1, 0] = mat[2]
    skewed_mat[1, 2] = -mat[0]
    skewed_mat[2, 0] = -mat[1]
    skewed_mat[2, 1] = mat[0]
    return skewed_mat


# Function to find the magnetic dipole moment
# Requires two instances of magpylib magnets to perform
def mag_moment(mag1, mag2):
    # Permeability in free space
    permeability = 4 * np.pi * (10 ** -7)

    # Distance between magnets 1 and 2
    mag1_to_mag2 = np.array((mag2.position - mag1.position) * 10 ** -3)
    # Magnitude of that distance
    norm_mag1_mag2 = np.sqrt(mag1_to_mag2[0] ** 2 + mag1_to_mag2[1] ** 2 + mag1_to_mag2[2] ** 2)
    # Noramlised vector between magnets 1 and 2
    normalised_mag1_to_mag2 = np.array(mag1_to_mag2 / norm_mag1_mag2)[np.newaxis]

    # Acquiring the different components
    matrix_const = np.linalg.inv(
        3 * np.matmul(np.transpose(normalised_mag1_to_mag2), normalised_mag1_to_mag2) - np.identity(3))
    non_matrix_const = ((4 * np.pi * norm_mag1_mag2 ** 3) / permeability) * mag1.getB(mag2.position) * 10 ** -3

    # Making the dipole moment
    dipole_moment = np.matmul(matrix_const, non_matrix_const)
    return dipole_moment


# Function to find the magnetic force
# Requires two instances of magpylib magnets to perform
def mag_force(EPM, IPM):
    # Permeability in free space
    permeability = 4 * np.pi * (10 ** -7)

    # Vector between EPM and IPM
    EPM_to_IPM = (IPM.position - EPM.position) * 10 ** -3
    # Magnitude of that vector
    norm_EPM_to_IPM = np.sqrt(EPM_to_IPM[0] ** 2 + EPM_to_IPM[1] ** 2 + EPM_to_IPM[2] ** 2)
    # Normalised vector
    normalised_EPM_to_IPM = np.array(EPM_to_IPM / norm_EPM_to_IPM)[np.newaxis]

    # Getting the dipole moments for EPM and IPM
    dipole_moment_EPM = np.array(mag_moment(s1, s2))[np.newaxis]
    dipole_moment_IPM = np.array(mag_moment(s2, s1))[np.newaxis]

    # Now to calculate the magnetic force attracting the IPM to the EPM
    const1 = np.matmul(normalised_EPM_to_IPM, dipole_moment_IPM.T) * dipole_moment_EPM.T
    const2 = np.matmul(normalised_EPM_to_IPM, dipole_moment_EPM.T) * dipole_moment_IPM.T
    const3 = np.matmul(dipole_moment_EPM, dipole_moment_IPM.T)

    const4_1 = np.matmul(normalised_EPM_to_IPM, dipole_moment_EPM.T)
    const4_2 = np.matmul(normalised_EPM_to_IPM, dipole_moment_IPM.T)
    const4 = 5 * np.matmul(const4_1, const4_2)

    # Combining the constants together gives us the magnetic force
    mag_force = ((3 * permeability) / (4 * np.pi * (norm_EPM_to_IPM ** 4))) * \
                (const1 + const2 + ((const3 - const4) * normalised_EPM_to_IPM.T))
    return mag_force


# Function to split up the paths into sub-paths and get the proper number of elements for beam initialisation
def lenghts_and_elements(all_paths):
    # Preparing the arrays to hold the path lengths and number of elements
    num_elem = []
    path_length = []
    for i in range(0, all_paths.shape[0] - 1):
        # Getting the length
        length = np.sqrt((all_paths[i][0] - all_paths[i + 1][0]) ** 2 + (all_paths[i][1] - all_paths[i + 1][1]) ** 2
                         + (all_paths[i][2] - all_paths[i + 1][2]) ** 2)

        # Slope was found analytically
        slope = (50 - 5) / (0.2 - 0.0154)

        # Saving the length and number of elements
        path_length.append(length)
        num_elem.append(round(slope * length, 0))
    return path_length, num_elem


# Compute beam position for sherable and unsherable beams.
def analytical_result(arg_rod, arg_end_force, shearing=True, n_elem=500):
    # Getting the length
    base_length = np.sum(arg_rod.rest_lengths)
    # Getting all the points between 0 and the end of the beam
    arg_s = np.linspace(0.0, base_length, n_elem)

    # Making sure to only get force applid in one axis
    if type(arg_end_force) is np.ndarray:
        acting_force = arg_end_force[np.nonzero(arg_end_force)]
    else:
        acting_force = arg_end_force

    # Needed to keep track of proper directionality
    acting_force = -acting_force

    # Get all the prefactor terms
    linear_prefactor = -acting_force / arg_rod.shear_matrix[0, 0, 0]
    quadratic_prefactor = -acting_force / 2.0 * np.sum(arg_rod.rest_lengths / arg_rod.bend_matrix[0, 0, 0])
    cubic_prefactor = (acting_force / 6.0) / arg_rod.bend_matrix[0, 0, 0]

    # Shearing and non-shearing have two different equations to them
    if shearing:
        return arg_s, arg_s * linear_prefactor + arg_s ** 2 * quadratic_prefactor + arg_s ** 3 * cubic_prefactor
    else:
        return arg_s, arg_s ** 2 * quadratic_prefactor + arg_s ** 3 * cubic_prefactor


# Function to define what kind of movements are needed (UP, DOWN, or LEVEL for the moment)
def define_movements(start, target):
    movements = []

    # changes in z - UP / DOWN / LEVEL
    if start[2] < target[2]:
        movements.append('UP')
    elif start[2] > target[2]:
        movements.append('DOWN')
    else:
        movements.append('LEVEL')

    # changes in y - LEFT / RIGHT / CENTER
    if start[1] < target[1]:
        movements.append('LEFT')
    elif start[1] > target[1]:
        movements.append('RIGHT')
    else:
        movements.append('CENTER')

    # Nothing in x as we assume it is always moving forward
    return movements


# Function to apply deformation of the beam with the applied force of the magnets and gravity
def Position_Post_Deformation(s1, s2, shearable_rod, num_elements, Force_syst):
    # Define variable which is the force acted to get the beam deformed to a certain position
    end_force = Force_syst

    # Get the position of the tip of the rod after it was bent
    analytical_shearable_position = np.empty((3, num_elements), dtype=float)

    # As the analytical results work only for a 2D solution, we need to go through the x, y and z elements
    # individually to get the final position of the rod in our simplified solution
    # x-element
    if end_force[0][0] != 0:
        temp = analytical_result(shearable_rod, np.transpose(np.array([0, 0, end_force[0][0]])[np.newaxis]),
                                 shearing=True, n_elem=num_elements)
        analytical_shearable_position[0][:] = temp[1][:]
    else:
        analytical_shearable_position[0][:] = shearable_rod.position_collection[0, :]

    # y-element
    if end_force[1][0] != 0:
        temp = analytical_result(shearable_rod, np.transpose(np.array([0, 0, end_force[1][0]])[np.newaxis]),
                                 shearing=True, n_elem=num_elements)
        analytical_shearable_position[1][:] = temp[1][:]
    else:
        analytical_shearable_position[1][:] = shearable_rod.position_collection[1, :]

    # z-element
    if end_force[2][0] != 0:
        temp = analytical_result(shearable_rod, np.transpose(np.array([0, 0, end_force[2][0]])[np.newaxis]),
                                 shearing=True, n_elem=num_elements)
        analytical_shearable_position[2][:] = temp[1][:]
    else:
        analytical_shearable_position[2][:] = shearable_rod.position_collection[2, :]

    # Set magnet to that new tip position
    vector_to_move = np.array(
        [analytical_shearable_position[0][num_elements - 1], analytical_shearable_position[1][num_elements - 1],
         analytical_shearable_position[2][num_elements - 1]]) * 10 ** 3 - s2.position
    s2.move(vector_to_move)
    s1.move(vector_to_move)

    return s1, s2, analytical_shearable_position


# Function to deform the beam to a specified position
def beam_deformation(s1, s2, shearable_rod, num_elements, mag_shape, rod_shape, rod_int_char, target, previous_force):
    analytical_shearable_positions = []
    target_position = np.array(target)

    # Saving the locations for the EPM and IPM
    EPM_locs = []
    IPM_locs = []

    # Defining the types of movements needed to complete the trajectory
    movements = define_movements([shearable_rod.position_collection[0, 0], shearable_rod.position_collection[1, 0],
                                  shearable_rod.position_collection[2, 0]], target_position)

    # Setting up gravity
    Gravity_beam = np.pi * rod_shape[0] ** 2 * rod_shape[1] * rod_int_char[0] * np.array([0, 0, -9.81])
    Gravity_mag = 0

    # Magnet gravity is dependent on the shape of the magnet
    if mag_shape == 'Sphere':
        Gravity_mag = ((4 / 3) * np.pi * ((s2.dimension * 10 ** -3) / 2) ** 3) * 7450 * np.array([0, 0, -9.81])
    elif mag_shape == 'Cylinder':
        Gravity_mag = ((np.pi * ((s2.dimension[0] * 10 ** -3) / 2) ** 2) * s2.dimension[
            1] * 10 ** -3) * 7450 * np.array([0, 0, -9.81])
    elif mag_shape == 'Box':
        Gravity_mag = (s2.dimension[0] * 10 ** -3 * s2.dimension[1] * 10 ** -3 * s2.dimension[
            2] * 10 ** -3) * 7450 * np.array([0, 0, -9.81])
    Gravity_syst = np.transpose(np.array(Gravity_beam + Gravity_mag)[np.newaxis])

    # First set of positions to save ## THE INITIAL FORCE WHICH DRIVES THE MOVEMENT ##
    s1, s2, analytical_shearable_position = \
        Position_Post_Deformation(s1, s2, shearable_rod, num_elements, previous_force)

    # First set of positions to save ## THE INITIAL POSITION ##
    EPM_locs.append(s1.position)
    IPM_locs.append(s2.position)

    # Saving the beam deformation
    analytical_shearable_positions.append(analytical_shearable_position)

    # Find the appropriate force needed for deformation
    s1, s2, Savepoint = find_appropriate_force(s1, s2, target_position, movements, Gravity_syst, shearable_rod,
                                               num_elements)

    # To be used to flip EPM down or up depending on situation
    # Solely used for DOWN only movements
    if (((s1.position[2] < s2.position[2]) and (Savepoint[2] >= s2.position[2])) or \
        ((s1.position[2] > s2.position[2]) and (Savepoint[2] <= s2.position[2]))) and ('CENTER' in movements):
        EPM_inter, EPM_Savepoint = inter_down(s1, s2, np.transpose(np.array([0, 0, 0])[np.newaxis]))
        # We don't have a deformation as we are assuming the EPM is switched fast enough so that we can create
        # equilibrium with gravity with the catheter in its post deformed state
        EPM_locs.append(EPM_inter)
        IPM_locs.append(s2.position)
        s1.setPosition(EPM_Savepoint)

        # Find the deformations
        _, _, analytical_shearable_position = \
            Position_Post_Deformation(s1, s2, shearable_rod, num_elements, previous_force)

        # Saving the beam deformation
        analytical_shearable_positions.append(analytical_shearable_position)

    # Getting the previous beam deformation with the magnets placed in the proper position to apply the new deformation
    _, _, analytical_shearable_position = \
        Position_Post_Deformation(s1, s2, shearable_rod, num_elements, previous_force)

    # First set of positions to save ## THE INITIAL FORCE WHICH DRIVES THE MOVEMENT ##
    EPM_locs.append(s1.position)
    IPM_locs.append(s2.position)

    # Saving the beam deformation
    analytical_shearable_positions.append(analytical_shearable_position)

    # Getting the total force applying to the tip of the catheter
    Syst_Force = Gravity_syst + mag_force(s1, s2)

    # Find the deformations
    s1, s2, analytical_shearable_position = \
        Position_Post_Deformation(s1, s2, shearable_rod, num_elements, Syst_Force)
    # Saving the beam deformation
    analytical_shearable_positions.append(analytical_shearable_position)

    ## Second set of positions to save ## THE DEFORMED IPM POSITION AND THE EPM POSITION WHICH KEEPS
    # THE REQUIRED FORCE FOR DEFORMATION ##
    EPM_locs.append(s1.position)
    IPM_locs.append(s2.position)

    return s1, s2, analytical_shearable_positions, EPM_locs, IPM_locs, movements, Syst_Force


# Function to deal with intermediary step in DOWN movements
def inter_down(s1, s2, Force, Side='LEFT'):
    tol = 10 ** -4
    iter = 1

    # Save the current EPM position as we don't want to permanently change EPM position
    EPM_savepoint = s1.position
    # Now we can set it to a different value so we can completely erase all trace of the magnets interacting
    if Side == 'LEFT':
        s1.setPosition([s2.position[0], s2.position[1] + 100, s2.position[2]])
    else:
        s1.setPosition([s2.position[0], s2.position[1] - 100, s2.position[2]])
    Mag_Force = mag_force(s1, s2)

    # While loop to ensure that we eventually get a solution which matches the desired criteria
    while ((abs(Mag_Force[1][0]) < abs(Force[1][0]) - tol) or (
            abs(Mag_Force[1][0]) > abs(Force[1][0]) + tol)) and iter < 50:

        # If too close
        if (abs(Mag_Force[1][0]) < abs(Force[1][0]) - tol):
            new_EPM_pos = (s1.position + s2.position) / 2
            s1.setPosition(new_EPM_pos)

        # If too far
        elif (abs(Mag_Force[1][0]) > abs(Force[1][0]) + tol):
            if Force[1][0] != 0:
                abs_diff = (abs(Mag_Force[1][0]) - abs(Force[1][0])) / abs(Force[1][0] * 100)
            else:
                abs_diff = 0.5
            s1.move((s1.position - s2.position) * abs_diff)

        Mag_Force = mag_force(s1, s2)
        iter += 1
    EPM_loc = s1.position
    return EPM_loc, EPM_savepoint


# This function will place the EPM in the right position with respect to the IPM in order to to generate the necessary
# amount of force for the desired deformation
def Force_equilib(s1, s2, Force):
    tol = 10 ** -9
    iter = 1
    Mag_Force = mag_force(s1, s2)

    # While loop to ensure that we eventually get a solution which matches the desired criteria
    while ((abs(Mag_Force[2][0]) < abs(Force[2][0]) - tol) or (
            abs(Mag_Force[2][0]) > abs(Force[2][0]) + tol)) and iter < 300:

        # If too close
        if (abs(Mag_Force[2][0]) < abs(Force[2][0]) - tol):
            new_EPM_pos = (s1.position + s2.position) / 2
            s1.setPosition(new_EPM_pos)

        # If too far
        elif (abs(Mag_Force[2][0]) > abs(Force[2][0]) + tol):
            if Force[2][0] != 0:
                abs_diff = (abs(Mag_Force[2][0]) - abs(Force[2][0])) / abs(Force[2][0] * 100)
            else:
                abs_diff = 0.5
            s1.move((s1.position - s2.position) * abs_diff)
        Mag_Force = mag_force(s1, s2)
        iter += 1
    return s1


# This function will be used in the interim step for determining left and right motion
def Force_equilib_y_coord(s1, s2, Force):
    tol = 10 ** -9
    iter = 1
    Mag_Force = mag_force(s1, s2)

    # While loop to ensure that we eventually get a solution which matches the desired criteria
    while ((abs(Mag_Force[1][0]) < Force - tol) or (abs(Mag_Force[1][0]) > Force + tol)) and iter < 300:

        # If too close
        if (abs(Mag_Force[1][0]) < Force - tol):
            new_EPM_pos = (s1.position + s2.position) / 2
            s1.setPosition(new_EPM_pos)

        # If too far
        elif (abs(Mag_Force[1][0]) > Force + tol):
            if Force != 0:
                abs_diff = (abs(Mag_Force[1][0]) - Force) / (Force * 100)
            else:
                abs_diff = 0.5
            s1.move((s1.position - s2.position) * abs_diff)
        Mag_Force = mag_force(s1, s2)

        iter += 1
    return s1


# Function which will do the force calculations necessary to find position of EPM / IPM which creates the force needed
# to reach a target
def find_appropriate_force(EPM, IPM, Target, Direction, Gravity_Force, Beam, n_elem, shearing=True):
    # First we need to calculate the amount of force needed to get to the desired postion
    # This will depend on the direction of the movements

    # We only will ever have forces in y and z to manipulate the catheter up and down, left or right
    # or staying level and center
    EPM_Savepoint = s1.position

    # For when we are just moving forward
    if ("CENTER" in Direction) and ("LEVEL" in Direction):
        # Just in case we are returning from a left / right positioning
        EPM.setPosition([EPM.position[0], IPM.position[1], EPM.position[2]])
        if EPM.position[2] < 0:
            EPM.setPosition([EPM.position[0], EPM.position[1], -EPM.position[2]])
        # Find the EPM position that allows for equilibrium with gravity
        EPM = Force_equilib(EPM, IPM, abs(Gravity_Force))

    # For when we are moving up and down
    if (("UP" in Direction) or ("DOWN" in Direction)) and ("CENTER" in Direction):
        # Just in case we are returning from a left / right positioning
        EPM.setPosition([EPM.position[0], IPM.position[1], EPM.position[2]])

        # We need to define the amount of force needed in the z axis in order to acquire the desired deformation
        # Based on Euler-Bernouilli formula
        base_length = np.sum(Beam.rest_lengths)
        arg_s = np.linspace(0.0, base_length, n_elem)

        # Get all the prefactors
        linear_prefactor = -1 / Beam.shear_matrix[0, 0, 0]
        quadratic_prefactor = -1 / 2.0 * np.sum(Beam.rest_lengths / Beam.bend_matrix[0, 0, 0])
        cubic_prefactor = (1 / 6.0) / Beam.bend_matrix[0, 0, 0]

        # Dependent on if we shear or not
        if shearing:
            Force_z = -Target[2] / (arg_s[-1] * linear_prefactor + arg_s[-1] ** 2 * quadratic_prefactor + arg_s[
                -1] ** 3 * cubic_prefactor)
        else:
            Force_z = -Target[2] / (arg_s[-1] ** 2 * quadratic_prefactor + arg_s[-1] ** 3 * cubic_prefactor)

        #  Finally get the force applied to the system to get the specified point
        Force_z = np.transpose(np.array([0, 0, Force_z])[np.newaxis])

        # Now that we acquire the force required to get the desired deformation
        Magnetic_Force_needed = Force_z - Gravity_Force

        ## ADDING CORRECTION TO DOWN MOVEMENT ##
        if Magnetic_Force_needed[2] < 0:
            EPM.setPosition([EPM.position[0], EPM.position[1], -EPM.position[2]])

        # Here is the position of the EPM to get the force needed for deformation
        EPM = Force_equilib(EPM, IPM, Magnetic_Force_needed)
        Acqu_Mag = mag_force(EPM, IPM)

        # Correction which exists to make sure that downwards motion does occur as is expected
        if (Acqu_Mag[2] >= 0) and (Magnetic_Force_needed[2] < 0):
            move_vect = 2 * np.array(IPM.position - EPM.position)
            EPM.move(move_vect)

    # This is for anything involving left or right movement
    if ("LEFT" in Direction) or ("RIGHT" in Direction):
        # We need to define the amount of force needed in the z axis in order to acquire the desired deformation
        # Based on Euler-Bernouilli formula
        base_length = np.sum(Beam.rest_lengths)
        arg_s = np.linspace(0.0, base_length, n_elem)

        # Get all the prefactors
        linear_prefactor = -1 / Beam.shear_matrix[0, 0, 0]
        quadratic_prefactor = -1 / 2.0 * np.sum(Beam.rest_lengths / Beam.bend_matrix[0, 0, 0])
        cubic_prefactor = (1 / 6.0) / Beam.bend_matrix[0, 0, 0]

        # Dependent on if we shear or not
        if shearing:
            Force_y = -Target[1] / (arg_s[-1] * linear_prefactor + arg_s[-1] ** 2 * quadratic_prefactor +
                                    arg_s[-1] ** 3 * cubic_prefactor)
            Force_z = -Target[2] / (arg_s[-1] * linear_prefactor + arg_s[-1] ** 2 * quadratic_prefactor + arg_s[
                -1] ** 3 * cubic_prefactor)
        else:
            Force_y = -Target[1] / (arg_s[-1] ** 2 * quadratic_prefactor + arg_s[-1] ** 3 * cubic_prefactor)
            Force_z = -Target[2] / (arg_s[-1] ** 2 * quadratic_prefactor + arg_s[-1] ** 3 * cubic_prefactor)

        # Finding magnitude of force needed
        Force_mag = np.sqrt(Force_y ** 2 + (Force_z - Gravity_Force[2]) ** 2)

        # Set the EPM directly parallel to IPM to get the force in y from IPM to EPM
        if 'RIGHT' in Direction:
            EPM.setPosition([IPM.position[0], IPM.position[1] - 100, IPM.position[2]])
        else:
            EPM.setPosition([IPM.position[0], IPM.position[1] + 100, IPM.position[2]])

        # Here we achieve the distance needed to acquire the magnitude Force_mag
        EPM = Force_equilib_y_coord(EPM, IPM, Force_mag)

        # Finding the correct angle
        # Knowing what my desired forces in y, z and magnitude are I can extract the angle that is required
        if (Force_z - Gravity_Force[2]) >= 0:
            desired_angle = np.arccos(Force_y / Force_mag) * (180 / np.pi)
        else:
            desired_angle = np.arcsin((Force_z - Gravity_Force[2]) / Force_mag) * (180 / np.pi)
        if 'RIGHT' in Direction:
            desired_angle = desired_angle + 180

        EPM.rotate(desired_angle, [1, 0, 0], IPM.position)

        # Acquire the magnetic force at the starting position, then enter the while loop
        force = mag_force(EPM, IPM)

        # To make sure that EPM is placed properly
        # N.B. this only works if the only fault is the signage
        if ((Force_z - Gravity_Force[2] < 0) and (force[2][0] > 0)) or \
                ((Force_z - Gravity_Force[2] > 0) and (force[2][0] < 0)):
            EPM.move([0, 0, 2 * (IPM.position[2] - EPM.position[2])])

        if ((Force_y < 0) and (force[1][0] > 0)) or ((Force_y > 0) and (force[1][0] < 0)):
            EPM.move([0, 2 * (IPM.position[1] - EPM.position[1]), 0])

    return EPM, IPM, EPM_Savepoint


# Defining a function that will create symmetrical UP/DOWN and LEFT/RIGHT paths
def symm_paths(X_val, Direction, Num_elem):
    Paths = []
    factor = abs(0.1/(Num_elem/2))
    # Down / Up
    if Direction == 'Vertical':
        for i in range(0, Num_elem+1):
            Paths.append([X_val, 0, -(i-Num_elem/2)*factor])

    # Left / Right
    elif Direction == 'Horizontal':
        for i in range(0, Num_elem + 1):
            Paths.append([X_val, (i-Num_elem/2)*factor, 0])

    # Up Left / Down Right
    elif Direction == 'UL/DR':
        for i in range(0, Num_elem + 1):
            Paths.append([X_val, (i-Num_elem/2)*factor, -(i-Num_elem/2)*factor])

    # Down Left / Up Right
    elif Direction == 'DL/UR':
        for i in range(0, Num_elem + 1):
            Paths.append([X_val, (i-Num_elem/2)*factor, (i-Num_elem/2)*factor])

    # Forward
    elif Direction == 'FWD':
        factor = abs(0.2 / Num_elem)
        for i in range(0, Num_elem + 1):
            Paths.append([i * factor, 0, 0])

    return np.array(Paths)


# Using this as a testing bed for catheter manipulation
if __name__ == "__main__":
    # Setting up the path to travel with the catheter
    ## SIMPLE PATHS TO TEST WHETHER THE SYSTEM WORKS OR NOT ##
    ## JUST UNCOMMENT THE ONE YOU WISH TO USE ##
    ## AND COMMENT OUT THE ONES YOU DON'T WISH TO USE ##
    # UP / DOWN / LEVEL / CENTER
    # testing_path = np.array([[0, 0, 0], [0.0154, 0, 0], [0.0308, 0, 0], [0.0462, 0, 0.004],
    #                          [0.0615, 0, 0.006], [0.0769, 0, 0.007],
    #                          [0.0923, 0, 0.006], [0.1077, 0, 0.004],
    #                          [0.1231, 0, 0], [0.1385, 0, -0.04],
    #                          [0.1538, 0, -0.04], [0.1692, 0, 0], [0.1846, 0, 0], [0.2, 0, 0]])

    # LEVEL / CENTER
    # testing_path = np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]])

    # UP
    # testing_path = np.array([[0, 0, 0], [0.1, 0, 0.01], [0.2, 0, 0.02]])

    # DOWN
    # testing_path = np.array([[0, 0, 0], [0.1, 0, -0.01], [0.2, 0, -0.02]])

    # LEFT
    # testing_path = np.array([[0, 0, 0], [0.1, 0.01, 0.0], [0.2, 0.02, 0.0]])

    # RIGHT
    # testing_path = np.array([[0, 0, 0], [0.1, -0.01, 0.0], [0.2, -0.02, 0.0]])

    ## PATHS USED FOR ANIMATIONS ##
    ## JUST UNCOMMENT THE ONE YOU WISH TO USE ##
    ## AND COMMENT OUT THE ONES YOU DON'T WISH TO USE ##
    # COMPLEX PATH
    testing_path = np.array([[0, 0, 0], [0.0154, 0, 0], [0.0308, 0, 0], [0.0462, 0, 0.004],
                             [0.0615, 0.004, 0.006], [0.0769, 0, 0.007],
                             [0.0923, -0.004, 0.006], [0.1077, 0, 0.004],
                             [0.1231, 0, 0], [0.1385, -0.04, -0.04],
                             [0.1538, 0.04, -0.04], [0.1692, 0, 0], [0.1846, 0, 0], [0.2, 0, 0]])

    # Symmetrical paths
    # testing_path = symm_paths(0.2, 'Vertical', 100)
    # testing_path = symm_paths(0.2, 'Horizontal', 100)
    # testing_path = symm_paths(0.2, 'UL/DR', 100)
    # testing_path = symm_paths(0.2, 'DL/UR', 100)
    # testing_path = symm_paths(0.2, 'FWD', 20)

    # Splitting the path into different sub paths which can be approximately modelled by beams
    _, elem = lenghts_and_elements(testing_path)

    # Magnet setup
    # Magnet position
    EPM_pos = [0, 0, 100]
    IPM_pos = [0, 0, 0]

    # EPM
    s1 = Sphere(mag=[1705, 0, 0], dim=40, pos=EPM_pos)
    # IPM
    s2 = Sphere(mag=[-1705, 0, 0], dim=2, pos=IPM_pos)

    # Need to save the magnet positions and orientations
    EPM_positions = []
    IPM_positions = []

    # Also need to save the beam positions
    Beam_positions = []

    # Movements
    Movements = []

    # Previous Force
    Prev_Force = np.transpose(np.array([0, 0, 0])[np.newaxis])
    Prev_EPM = [0, 0, 0]
    Prev_IPM = [0, 0, 0]

    # To help set magnet positions and define the rods
    cumulative_elements = 0
    cumulative_length = 0

    for i in range(0, len(elem)):
        # Setting up beam parameters
        if testing_path[i+1][0] > testing_path[i][0]:
            cumulative_elements += int(elem[i])
        else:
            cumulative_elements = 50

        # Beam parameters
        n_elem = cumulative_elements
        normal = np.array([0.0, 1.0, 0.0])
        direction = np.array([1.0, 0.0, 0.0])
        density = 1700
        nu = 0.25
        E = 2e9
        poisson_ratio = 0.48
        start = np.array([0.0, 0.0, 0.0])
        base_length = np.sqrt(0**2 + testing_path[i + 1][0]**2)
        base_radius = 0.001
        base_area = np.pi * base_radius ** 2
        target = testing_path[i + 1]


        # Creating the beam
        shearable_rod = CosseratRod.straight_rod(n_elem, start, direction, normal, base_length, base_radius, density,
                                                 nu, E, poisson_ratio, )

        # Setting the magnets in appropriate positions
        if (Prev_Force[1][0] == 0) and (Prev_Force[2][0] == 0):
            s2.setPosition(np.array([shearable_rod.position_collection[0, n_elem], shearable_rod.position_collection[1, n_elem], shearable_rod.position_collection[2, n_elem]]) * 10 ** 3)
            s1.setPosition(s2.position + np.array([0, 0, 100]))
        else:
            s2.setPosition(np.array([shearable_rod.position_collection[0, n_elem]* 10 ** 3, Prev_IPM[1], Prev_IPM[2]]))
            s1.setPosition(np.array([shearable_rod.position_collection[0, n_elem]* 10 ** 3, Prev_EPM[1], Prev_EPM[2]]))

        # Moving the magnets and getting the
        s1, s2, analytical_shearable_position, EPM_locs, IPM_locs, movements, Produced_Force = \
            beam_deformation(s1, s2, shearable_rod, n_elem + 1, 'Sphere', [base_radius, base_length], [density, E], target, Prev_Force)

        # Saving the previous force for future use
        Prev_Force = Produced_Force

        # Saving the previous EPM and IPM positions for future use
        EPM = np.array(EPM_locs) * 10 ** -3
        IPM = np.array(IPM_locs) * 10 ** -3
        Prev_EPM = EPM[-1] * 10**3
        Prev_IPM = IPM[-1] * 10**3

        # Save all the the needed variables / values
        EPM_positions.append(EPM)
        IPM_positions.append(IPM)
        Beam_positions.append(analytical_shearable_position)
        Movements.append(movements)

    # Function to plot the magnet going through a path as well as the resulting bent beam
    def plot_timoshenko(s1, s2, Beam_positions, EPM_positions, IPM_positions):
        import matplotlib.pyplot as plt

        # create figure
        fig = plt.figure(figsize=(9, 5))
        ax1 = fig.add_subplot(121, projection='3d')  # 3D-axis
        ax2 = fig.add_subplot(122, projection='3d')  # 2D-axis
        ax2.grid(b=True, which="major", color="grey", linestyle="-", linewidth=0.25)

        # Go over every target point
        for i in range(0, len(EPM_positions)):
            # Helps seperate based on if there were 2 intermediate points or 1
            if (np.array(EPM_positions[i]).shape[0] == 3):
                # If one intermediate point
                lim = 3
            else:
                # If two intermediate points
                lim = 4

            # For every recorded magnet position
            for j in range(0, lim):
                # Set the x, y and z values of the EPM and IPM
                EPM_x = EPM_positions[i][j][0]
                EPM_y = EPM_positions[i][j][1]
                EPM_z = EPM_positions[i][j][2]
                IPM_x = IPM_positions[i][j][0]
                IPM_y = IPM_positions[i][j][1]
                IPM_z = IPM_positions[i][j][2]

                # Set the magpylib magnets to the respective positions
                s1.setPosition(EPM_positions[i][j] * 10 ** 3)
                s2.setPosition(IPM_positions[i][j] * 10 ** 3)

                # Make a collection out of both magnets
                c = Collection(s1, s2)

                # display system geometry on ax1
                # This will display the physical magnets
                displaySystem(c, subplotAx=ax1, suppress=True)

                # This is were we display the magnet positions
                # If only 1 intermediate point
                if lim == 3:
                    # Starting position to generate force
                    if (i == len(EPM_positions) - 1) and (j == 0):
                        ax2.scatter(EPM_x, EPM_y, EPM_z, marker="*", c="g", label="EPM start")
                        ax2.scatter(IPM_x, IPM_y, IPM_z, marker="o", c="r", label="IPM start")

                    # Intermediary position to get IPM up to deformation point
                    elif (i == len(EPM_positions) - 1) and (j == 1):
                        ax2.scatter(EPM_x, EPM_y, EPM_z, marker="*", c="c", label="EPM inter")
                        ax2.scatter(IPM_x, IPM_y, IPM_z, marker="o", c="y", label="IPM inter")

                    # Keep the IPM at the desired point
                    elif (i == len(EPM_positions) - 1) and (j == 2):
                        ax2.scatter(EPM_x, EPM_y, EPM_z, marker="*", c="b", label="EPM final")
                        ax2.scatter(IPM_x, IPM_y, IPM_z, marker="o", c="m", label="IPM final")

                    # Starting position to generate force
                    elif (i != len(EPM_positions) - 1) and (j == 0):
                        ax2.scatter(EPM_x, EPM_y, EPM_z, marker="*", c="g")
                        ax2.scatter(IPM_x, IPM_y, IPM_z, marker="o", c="r")

                    # Intermediary position to get IPM up to deformation point
                    elif (i != len(EPM_positions) - 1) and (j == 1):
                        ax2.scatter(EPM_x, EPM_y, EPM_z, marker="*", c="c")
                        ax2.scatter(IPM_x, IPM_y, IPM_z, marker="o", c="y")

                    # Keep the IPM at the desired point
                    elif (i != len(EPM_positions) - 1) and (j == 2):
                        ax2.scatter(EPM_x, EPM_y, EPM_z, marker="*", c="b")
                        ax2.scatter(IPM_x, IPM_y, IPM_z, marker="o", c="m")

                # If two intermediate points
                elif lim == 4:
                    # Starting position to generate force
                    if (i == len(EPM_positions) - 1) and (j == 0):
                        ax2.scatter(EPM_x, EPM_y, EPM_z, marker="*", c="g", label="EPM start")
                        ax2.scatter(IPM_x, IPM_y, IPM_z, marker="o", c="r", label="IPM start")

                    # Intermediary position to get IPM up to deformation point
                    elif (i == len(EPM_positions) - 1) and (j == 1):
                        ax2.scatter(EPM_x, EPM_y, EPM_z, marker="*", c="c", label="EPM inter")
                        ax2.scatter(IPM_x, IPM_y, IPM_z, marker="o", c="y", label="IPM inter")

                    # Intermediary position to get IPM up to deformation point
                    elif (i == len(EPM_positions) - 1) and (j == 2):
                        ax2.scatter(EPM_x, EPM_y, EPM_z, marker="*", c="c", label="EPM inter")
                        ax2.scatter(IPM_x, IPM_y, IPM_z, marker="o", c="y", label="IPM inter")

                    # Keep the IPM at the desired point
                    elif (i == len(EPM_positions) - 1) and (j == 3):
                        ax2.scatter(EPM_x, EPM_y, EPM_z, marker="*", c="b", label="EPM final")
                        ax2.scatter(IPM_x, IPM_y, IPM_z, marker="o", c="m", label="IPM final")

                    # Starting position to generate force
                    elif (i != len(EPM_positions) - 1) and (j == 0):
                        ax2.scatter(EPM_x, EPM_y, EPM_z, marker="*", c="g")
                        ax2.scatter(IPM_x, IPM_y, IPM_z, marker="o", c="r")

                    # Intermediary position to get IPM up to deformation point
                    elif (i != len(EPM_positions) - 1) and (j == 1):
                        ax2.scatter(EPM_x, EPM_y, EPM_z, marker="*", c="c")
                        ax2.scatter(IPM_x, IPM_y, IPM_z, marker="o", c="y")

                    # Intermediary position to get IPM up to deformation point
                    elif (i != len(EPM_positions) - 1) and (j == 2):
                        ax2.scatter(EPM_x, EPM_y, EPM_z, marker="*", c="c")
                        ax2.scatter(IPM_x, IPM_y, IPM_z, marker="o", c="y")

                    # Keep the IPM at the desired point
                    elif (i != len(EPM_positions) - 1) and (j == 3):
                        ax2.scatter(EPM_x, EPM_y, EPM_z, marker="*", c="b")
                        ax2.scatter(IPM_x, IPM_y, IPM_z, marker="o", c="m")

            # Get the x, y and z positions of the beam pots deformation
            Beam_x = Beam_positions[i][lim - 1][0]
            Beam_y = Beam_positions[i][lim - 1][1]
            Beam_z = Beam_positions[i][lim - 1][2]

            # Plot the deformed beam
            if i == len(EPM_positions) - 1:
                ax2.plot(Beam_x, Beam_y, Beam_z, "k-", label="Bent Beam")
            else:
                ax2.plot(Beam_x, Beam_y, Beam_z, "k-")

        # Put legends onto the final plot
        ax2.legend(prop={"size": 10}, loc="lower left")
        ax2.set_xlabel('X Position (m)', fontsize=12)
        ax2.set_ylabel('Y Position (m)', fontsize=12)
        ax2.set_zlabel('Z Position (m)', fontsize=12)
        plt.show()

    # From the list of EPM, IPM and Beam position lists get them in an order that is more legible
    def acquire_all_positions(EPM, IPM, Beam):
        EPM_corr = []
        IPM_corr = []
        Beam_corr = []
        for i in range(0, len(EPM)):
            for j in range(0, (EPM[i]).shape[0]):
                EPM_corr.append(EPM[i][j])
                IPM_corr.append(IPM[i][j])
                Beam_corr.append(Beam[i][j])
        return EPM_corr, IPM_corr, Beam_corr

    # Plot the entire path undertaken by the magnets and beam to reach the acquired pathway
    plot_timoshenko(s1, s2, Beam_positions, EPM_positions, IPM_positions)

    # Performing simplification of EPM, IPM and Beam positioning
    EPM_corr, IPM_corr, Beam_corr = acquire_all_positions(EPM_positions, IPM_positions, Beam_positions)

    # Start plotting for animation purposes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    data_skip = 0

    # Defining the initialisation function
    def init_func():
        ax.clear()
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_zlabel('Z Position (m)', fontsize=12)

    # Defining the updating function (This one only shows the beam and magnet positioning, without the magnet shapes)
    ## JUST UNCOMMENT THE ONE YOU WISH TO USE ##
    ## AND COMMENT OUT THE ONES YOU DON'T WISH TO USE ##
    def update_plot(i):
        ax.cla()
        ax.scatter(EPM_corr[i][0], EPM_corr[i][1], EPM_corr[i][2], marker="o", c="b", label="EPM")
        ax.scatter(IPM_corr[i][0], IPM_corr[i][1], IPM_corr[i][2], marker="o", c="r", label="IPM")
        ax.plot3D(Beam_corr[i][0], Beam_corr[i][1], Beam_corr[i][2], "k-", label="Beam")
        plt.xlim((0.0, 0.2))
        plt.ylim((-0.2, 0.2))
        ax.set_zlim(-0.2, 0.2)
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_zlabel('Z Position (m)', fontsize=12)
        plt.legend(prop={"size": 10}, loc="lower left")

    # Defining the updating function (This one only shows the beam and magnet positioning, with the magnet shapes)
    # def update_plot(i):
    #     ax.cla()
    #     s1.setPosition(EPM_corr[i] * 10 ** 3)
    #     s2.setPosition(IPM_corr[i] * 10 ** 3)
    #
    #     # Make a collection out of both magnets
    #     c = Collection(s1, s2)
    #
    #     # display system geometry on ax1
    #     displaySystem(c, subplotAx=ax, suppress=True)
    #
    #     ax.scatter(EPM_corr[i][0]* 10 ** 3, EPM_corr[i][1]* 10 ** 3, EPM_corr[i][2]* 10 ** 3, marker="o", c="r", label="EPM")
    #     ax.scatter(IPM_corr[i][0]* 10 ** 3, IPM_corr[i][1]* 10 ** 3, IPM_corr[i][2]* 10 ** 3, marker="o", c="b", label="IPM")
    #     ax.plot3D(Beam_corr[i][0]* 10 ** 3, Beam_corr[i][1]* 10 ** 3, Beam_corr[i][2]* 10 ** 3, "k-", label="Beam")
    #     # plt.xlim((0.0, 0.2 * 10 ** 3))
    #     # plt.ylim((-0.2 * 10 ** 3, 0.2 * 10 ** 3))
    #     ax.set_xlim(-0.2 * 10 ** 3, 0.2 * 10 ** 3)
    #     ax.set_ylim(-0.2 * 10 ** 3, 0.2 * 10 ** 3)
    #     ax.set_zlim(-0.2 * 10 ** 3, 0.2 * 10 ** 3)
    #     plt.legend(prop={"size": 10}, loc="lower left")

    # Create, write and save the animation as a .gif file
    anim = FuncAnimation(fig,
                         update_plot,
                         frames=np.arange(0, len(EPM_corr)),
                         init_func=init_func,
                         interval=200)
    writergif = PillowWriter(fps=10)

    anim.save("Catheter_Moving.gif", writer=writergif)