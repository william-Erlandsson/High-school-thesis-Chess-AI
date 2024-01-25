!pip install tensorflow
!pip install chess


# Necessary libraries
import tensorflow
import chess
import matplotlib.pyplot as plt
import numpy as np
import time
import os


# Lists and Arrays

# The board
board = chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')

# Boolean lists showing the locations for each unique piece
w_pawn   =   np.asarray(board.pieces(chess.PAWN, chess.WHITE).tolist())
w_rook   =   np.asarray(board.pieces(chess.ROOK, chess.WHITE).tolist())
w_knight =   np.asarray(board.pieces(chess.KNIGHT, chess.WHITE).tolist())
w_bishop =   np.asarray(board.pieces(chess.BISHOP, chess.WHITE).tolist())
w_queen  =   np.asarray(board.pieces(chess.QUEEN, chess.WHITE).tolist())
w_king   =   np.asarray(board.pieces(chess.KING, chess.WHITE).tolist())
b_pawn   =   np.asarray(board.pieces(chess.PAWN, chess.BLACK).tolist())
b_rook   =   np.asarray(board.pieces(chess.ROOK, chess.BLACK).tolist())
b_knight =   np.asarray(board.pieces(chess.KNIGHT, chess.BLACK).tolist())
b_bishop =   np.asarray(board.pieces(chess.BISHOP, chess.BLACK).tolist())
b_queen  =   np.asarray(board.pieces(chess.QUEEN, chess.BLACK).tolist())
b_king   =   np.asarray(board.pieces(chess.KING, chess.BLACK).tolist())


# Single indice list for determining the player to move
turn   = [board.turn]


# List of booleans for the rooks ability to castle
castle = [
    bool(board.castling_rights & chess.BB_H1), 
    bool(board.castling_rights & chess.BB_A1), 
    bool(board.castling_rights & chess.BB_H8), 
    bool(board.castling_rights & chess.BB_A8)
         ]


# Bitboard for en passant squares
ep_square =  np.zeros(64)


# Adds a 1 to the ep_square array to show possible en passants locations
if isinstance(board.ep_square, int):
    np.put(ep_square, board.ep_square, 1)


# Bitboard

bitboard = np.concatenate((w_pawn.astype(int),   w_rook.astype(int),  w_knight.astype(int), 
                           w_bishop.astype(int), w_queen.astype(int), w_king.astype(int), 
                           b_pawn.astype(int),   b_rook.astype(int),  b_knight.astype(int), 
                           b_bishop.astype(int), b_queen.astype(int), b_king.astype(int),
                           turn,                 castle,              ep_square
                         ))
