
# coding: utf-8

# In[8]:

"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


# In[9]:

def custom_score(game, player): # survival_improved_score
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    
    # Take into account the number of moves available to user and opponent
    # Weight the number of move available to the opponent more to play more aggressively.
    
    if game.is_winner(player):
        return float("inf")
    
    if game.is_loser(player):
        return float("-inf")

    my_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    factor = 2
    
    return float(factor*my_moves - opp_moves)


# In[10]:

def custom_score_2(game, player): # proximinity_bias
    """Calculate the heuristic value of finding the maximum distance between
    the opponent's position and my position.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    
    # Try to position the user as close as possible to the opponent
    
    if game.is_winner(player):
        return float("inf")
    
    if game.is_loser(player):
        return float("-inf")

    my_location = game.get_player_location(player)
    opp_location = game.get_player_location(game.get_opponent(player))
    
    x = my_location[0] - opp_location[0]
    y = my_location[1] - opp_location[1]
    
    return 100 - float(x + y)


# In[11]:

def custom_score_3(game, player): # center_proximinity_bias
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    
    # Try to position the user between the opponent and the center of the board
    
    if game.is_winner(player):
        return float("inf")
    
    if game.is_loser(player):
        return float("-inf")
    
    my_location = game.get_player_location(player)
    center_board = int((game.width - 1) / 2), int((game.height - 1) / 2)
    opp_location = game.get_player_location(game.get_opponent(player))
      
    dist_from_center = my_location[0] - center_board[0], my_location[1] - center_board[1]
    dist_from_opp = my_location[0] - opp_location[0], my_location[1] - opp_location[1]
    
    x = dist_from_center[0] + dist_from_opp[0]
    y = dist_from_center[1] + dist_from_opp[1]
    
    return 100 - float(x + y)


# In[12]:

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.
    ********************  DO NOT MODIFY THIS CLASS  ********************
    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)
    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.
    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


# In[13]:

class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************
        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move
    
    def max_value(self, game, depth):
        #check for timeout
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Get legal moves 
        legal_moves = game.get_legal_moves()

        # If the top root or no other possible move, return the score
        if not legal_moves:
            return game.utility(self)

        if depth <= 0:
            return self.score(game, self)

        score = float("-inf")
        for move in legal_moves:
            next_state = game.forecast_move(move)
            score = max(score, self.min_value(next_state, depth - 1))
        return score
            
    def min_value(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return game.utility(self)

        if depth <= 0:
            return self.score(game, self)

        score = float("inf")
        for move in legal_moves:
            next_state = game.forecast_move(move)
            score = min(score, self.max_value(next_state, depth - 1))
        return score

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        
        # TODO: finish this function!
        
        #check for timeout
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return game.utility(self)

        if depth <= 0:
            return self.score(game, self)
            
        min_score = float("-inf")
        best_move = None
        
        for move in legal_moves:
            next_state = game.forecast_move(move)
            score = self.min_value(next_state, depth - 1)
            if score >= min_score:
                min_score, best_move = score, move
        
        return best_move


# In[7]:

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """
    
    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        
        self.time_left = time_left

        # TODO: finish this function!
        
        best_move = (-1, -1)
        depth = 1
        
        try:
            while True:
                best_move = self.alphabeta(game, depth)
                if self.time_left() <= 0:
                    raise SearchTimeout()
                depth += 1
        except SearchTimeout:
            pass
        return best_move
    
        
    def max_alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        
        #check for timeout
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return game.utility(self)

        if depth <= 0:
            return self.score(game, self)
        
        score = float("-inf")
        
        for move in legal_moves:
            next_state = game.forecast_move(move)
            score = max(score, self.min_alphabeta(next_state, depth - 1, alpha, beta))
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return score    
    
    def min_alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return game.utility(self)

        if depth <= 0:
            return self.score(game, self)
        
        score = float("inf")
        
        for move in legal_moves:
            next_state = game.forecast_move(move)
            score = min(score, self.max_alphabeta(next_state, depth - 1, alpha, beta))
            if score <= alpha:
                return score
            beta = min(beta, score)
        return score

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        # TODO: finish this function!
        
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return game.utility(self)

        if depth <= 0:
            return self.score(game, self)
        
        best_score = float("-inf")
        best_move = None
        
        for move in legal_moves:
            next_state = game.forecast_move(move)
            score = self.min_alphabeta(next_state, depth - 1, alpha, beta)
            if score >= best_score:
                best_score, best_move = score, move
            alpha = max(alpha, score)

        return best_move


# In[ ]:



