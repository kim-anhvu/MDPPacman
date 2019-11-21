# mdpAgents.py
# parsons/20-nov-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
from operator import itemgetter
import api
import random
import game
import util
import numbers
import copy


class Board:
    """
        This is a class for recreating the game layout using 2D array.
        Attributes:
            width (int): The width of the layout board.
            height (int): The height of the layout board.
            board ([[int][int]]): 2D array which each cell will hold a reward/utility.
    """

    def __init__(self, width, height, default_reward):
        """
            The constructor for the Board class.
            Parameters:
               width (int): The width of the layout board.
               height (int): The height of the layout board.
               board ([int][int]): 2D array which each cell will hold a reward/utility.
                                   Populates every grid with default_reward initially.
        """
        self.__width = width
        self.__height = height
        self.board = [[default_reward for x in range(self.__width)] for y in range(self.__height)]

    def get_board_width(self):
        """
            The function to get width of the board/layout.
            Parameters:
                None
            Returns:
                int: A number that represent number of columns of the board.
        """
        return self.__width

    def get_board_height(self):
        """
            The function to get height of the board/layout.
            Parameters:
                None
            Returns:
                int: A number that represent number of rows of the board.
        """
        return self.__height

    def __setitem__(self, (row, col), value):
        """
            The function to set a value to a specific position on the board.
            Parameters:
                row (int): Row number of chosen position.
                col (int): Column number of chosen position.
                value (int): Number to set to the position.
            Returns:
                None
        """
        self.board[row][col] = value

    def __getitem__(self, (row, col)):
        """
            The function to get a value at a specific position on the board.
            Parameters:
                row (int): Row number of chosen position.
                col (int): Column number of chosen position.
            Returns:
                int: Value stored at specified position.
        """

        return self.board[row][col]

    def convert_y(self, y):
        """
            The board's (represented using nested lists) 'y' coordinates start from the top
            and end at the bottom. So the top row is row 0. This is the opposite for the
            actual layout grid. This function is to be able to convert the y coordinate from
            a board to the y coordinate on the layout grid.
            Parameters:
                y (int): y value of board/grid layout

            Returns:
                int: converted y value of grid layout/board
        """
        return self.__height - 1 - y

    def set_position_values(self, positions, value):
        """
            The function to set a value to each position from list of positions.
            Parameters:
                positions ([(int, int)]): List of positions used to assign value to each
                                          position.
                value (int): Number to set to each position.
            Returns:
                None
        """
        for position in positions:
            self.board[int(self.convert_y(position[1]))][int(position[0])] = value


class MDPAgent(Agent):

    """
        This is a class to control Pacman using a MDP-solver.
    """
  # Constructor: this gets run when we first invoke pacman.py

    def __init__(self):
        name = "Pacman"

    def create_board(self, width, height, default_reward):
        """
            The function to create a Board object.
            Parameters:
                width (int): To create a board of this width.
                height (int): To create a board of this height.
                default_reward (float): Chosen number to initially populate every cell
                                        of the board.
            Returns:
                Board: A board of the width and height specified, which is populated with
                       the default_reward.
        """
        return Board(width, height, default_reward)

    def calculate_expected_utility(self, state, board, row, col):
        """
            The function to calculate expected utility for a specified position on the
            board.
            Parameters:
                board (Board): Chosen board to use.
                row (int): Row number of chosen position.
                col (int): Column number of chosen position.
            Returns:
                [(float, Directions)]: List of tuples of size 4, which contains a
                                       utility value and its corresponding action.
        """

        coord = [(col, row - 1), (col + 1, row), (col, row + 1),
                 (col - 1, row)]  # up, right, down, left
        prob = [(0, Directions.NORTH), (0, Directions.EAST),
                (0, Directions.SOUTH), (0, Directions.WEST)]

        for i, (c, r) in enumerate(coord):
            # Loop through coord list to check if it is not a wall. If it is,
            # replace the position with the current position.
            if not isinstance(board[r, c], numbers.Number):
                coord[i] = (col, row)
            # For each element in coord, pair element with corresponding reward in a tuple.
            coord[i] = (coord[i], board[coord[i][1], coord[i][0]])
        # probably delete this

        for i in range(len(coord)):
            # Algorithm that loops through coord to identify positions that are at a
            # right angle to current position and multiplies the corresponding
            # reward/utility to the right probability.
            aclockwise_right_angle = (i + len(coord) - 1) % len(coord)
            clockwise_right_angle = (i + 1) % len(coord)

            prob[i] = ((0.8 * coord[i][1]) +
                       (0.1 * coord[clockwise_right_angle][1]) +
                       (0.1 * coord[aclockwise_right_angle][1]), prob[i][1])
        return prob

    def value_iteration(self, state, board):
        """
            The function to carry out value iteration.
            Parameters:
                board (Board): Chosen board to use.
            Returns:
                Board: Board with updated utility values stored in each cell of board.
        """
        board_copy = copy.deepcopy(board)
        # Positions where value stored should not be altered.
        protected_pos = api.ghosts(state) + api.walls(state)
        gamma = 0.9     # discount value
        iterations = 7     # max number of iterations
        threshold = 0.01
        height = board_copy.get_board_height()
        width = board_copy.get_board_width()

        while iterations > 0:
            U = copy.deepcopy(board_copy)
            # total differences between previous board and new board which has been made at the end of the iteration
            total_difference = 0

            for row in range(height):
                for col in range(width):
                    value = board_copy[row, col]
                    # Check to make sure this position is not where a wall or a ghost is.
                    if (col, board.convert_y(row)) not in protected_pos:
                        # just take the utility from the list returned by calculate_expected_utility
                        expected_utility = [utility[0] for utility in self.calculate_expected_utility(
                            state, U, row, col)]
                        max_expected_utility = max(expected_utility)
                        board_copy[row, col] = board[row, col] + gamma * \
                            max_expected_utility  # Bellman's equation

            # calculate differences for each position using the old board (U) and new board (board_copy)
            # for row in range(height):
            #     for col in range(width):
            #         value = board_copy[row, col]
            #         if (col, board.convert_y(row)) not in protected_pos:
            #             total_difference += abs(round(value - U[row, col], 4))
            #
            # if total_difference <= threshold:
            #     break

            iterations -= 1

        return board_copy

    def getAction(self, state):
        """
            The function to work out next intended action carried out.
            Parameters:
                None
            Returns:
                Directions: Intended action that Pacman will carry out.
        """
        current_pos = api.whereAmI(state)
        corners = api.corners(state)
        food = api.food(state)
        ghosts = api.ghosts(state)
        walls = api.walls(state)
        legal = api.legalActions(state)
        capsules = api.capsules(state)

        width = max(corners)[0] + 1  # max x coordinate + 1
        height = max(corners, key=itemgetter(1))[1] + 1  # max y coordinate + 1

        board = self.create_board(width, height, -0.04)
        board.set_position_values(walls, 'x')
        board.set_position_values(capsules, 2)
        board.set_position_values(food, 1)
        board.set_position_values(ghosts, -3)

        # rewards of ghosts, walls and current position cannot be overridden
        protected_pos = set(ghosts + walls + [current_pos])

        # make sure all ghost coordinates are ints rather than floats
        int_ghosts = [(int(x), int(y)) for x, y in ghosts]

        for x, y in int_ghosts:
            # possible x coordinates that positions around the ghost can have
            x_coordinates = [x - 1, x, x + 1]
            # possible y coordinates that positions around the ghost can have
            y_coordinates = [y - 1, y, y + 1]

            for x_coord in x_coordinates:
                for y_coord in y_coordinates:
                    if (x_coord, y_coord) not in protected_pos:
                        # set the reward value to -2.
                        board[int(board.convert_y(y_coord)), int(x_coord)] = -2

        board = self.value_iteration(state, board)

        expected_utility = self.calculate_expected_utility(
            state, board, board.convert_y(current_pos[1]), current_pos[0])
        # returns action associated to the max utility out of all the legal actions.
        return api.makeMove(max([(utility, action) for utility, action in expected_utility if action in legal])[1], legal)

# Try to change ghost reward when capsule has been eaten
# Try to return Direction.STOP if best action is not in legal
