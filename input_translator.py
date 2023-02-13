import numpy as np
from game import unit_vector, find_angle, range_pi_to_pi
from numpy.linalg import norm


def soonest_to_hit(player, asteroids, NUMBER_OF_SECTORS):
    # Finds the soonest to hit asteroid (or just an edge) in each sector and returns:
    # 1. 1/(time until impact considering only relative radial speed) of the asteroid to the player
    # 2. the angle between the asteroids' relative velocity and the line joining the asteroid and the player (in the range -pi to pi)
    # 3. the angle of the asteroid in the frame of reference of the player (in the range -pi to pi)

    # Finds the asteroid with the highest 1/(time until impact) in each sector
    soonest_to_hit_asteroids = np.array([(0, np.pi, 0) for _ in range(NUMBER_OF_SECTORS)])
    for asteroid in asteroids:
        # Find the time until impact given relative radial speed of the asteroid and the player
        relative_velocity = asteroid.velocity - player.velocity
        displacement = asteroid.position - player.position
        # Positive radial speed means the asteroid is travelling away from the player
        radial_speed = np.dot(unit_vector(find_angle(displacement)), relative_velocity)
        # Find 1/(time until impact)
        inverse_time_to_impact = -radial_speed / norm(displacement)

        # Find the angle of the asteroid in the frame of reference of the player
        relative_angle = player.angle_player_object(asteroid.position)

        # Determine which sector the asteroid is in
        sector_number = 0
        # Should sector edges be inside the function or outside??
        SECTOR_EDGES = np.linspace(-np.pi, np.pi, NUMBER_OF_SECTORS + 1)
        while relative_angle > SECTOR_EDGES[sector_number + 1]:
            sector_number += 1
        # Check which 1/(time until impact) is highest in this sector
        if soonest_to_hit_asteroids[sector_number][0] < inverse_time_to_impact:
            # Find the angle between the asteroids' relative velocity and the line joining the asteroid and the player
            angle_relative_velocity = range_pi_to_pi(find_angle(relative_velocity) - find_angle(-displacement))
            soonest_to_hit_asteroids[sector_number] = (inverse_time_to_impact, angle_relative_velocity, relative_angle)
    return soonest_to_hit_asteroids


def input_translator(player, asteroids, NUMBER_OF_SECTORS, include_edges=False):
    inputs = np.zeros(NUMBER_OF_SECTORS * 3 + 2 * include_edges)
    inputs[:NUMBER_OF_SECTORS * 3] = soonest_to_hit(player, asteroids, NUMBER_OF_SECTORS).flatten()

    if include_edges == True:
        # Distance to nearest edge and the angle of the nearest point on the edge in the frame of reference of the player
        edge_point = player.nearest_edge()
        inputs[-2] = norm(edge_point - player.position)
        inputs[-1] = player.angle_player_object(edge_point)

    return np.array([inputs])
