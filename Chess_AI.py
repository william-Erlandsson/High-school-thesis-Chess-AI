!pip install tensorflow
!pip install chess
!pip install pandas
!pip install XLRD


# Libraries
import chess
import numpy as np
import pandas as pd
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.utils as utils
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.regularizers as regularizers
import tensorflow.keras.callbacks as callbacks
from tensorflow import convert_to_tensor




# Bitboard
# dictionary to convert letters to integers
squares_index = {
    'a' : 0,
    'b' : 1,
    'c' : 2,
    'd' : 3,
    'e' : 4,
    'f' : 5,
    'g' : 6,
    'h' : 7
}




def square_to_index(square):
    """square_to_index converts any given square on a chess board into an corresponding integer for 
    use with the chess library"""
    letter = chess.square_name(square)
    return 8- int(letter[1]), squares_index[letter[0]]



def bitboard(board):
    """bitboard takes the inputed board position expressed in Forsyth-Edwards Notation (aka FEN) and 
    converts it to a (14, 8, 8) bitboard that contains the matrixes for both the individual pieces aswell as 
    the legal moves for white's and black's pieces in the order (white_pawn, white_knight, white_bishop, 
    white_rook, white_queen, white_king followed by matrixes for black's pieces in the same order and lastly 
    white's legal moves and then black's legal moves.)"""

    board = chess.Board(board)
    board_matrix = np.zeros((14, 8, 8), dtype=np.int8)

    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            idx = np.unravel_index(square, (8, 8))
            board_matrix[piece - 1][7 - idx[0]][idx[1]] = 1
        
        for square in board.pieces(piece, chess.BLACK):
            idx = np.unravel_index(square, (8, 8))
            board_matrix[piece + 5][7 - idx[0]][idx[1]] = 1


    turn = board.turn
    board.turn = chess.WHITE

    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board_matrix[12][i][j] = 1
    
    board.turn = chess.BLACK
    
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board_matrix[13][i][j] = 1
    
    board.turn = turn


    return board_matrix




# Dataset conversion
data = pd.read_excel('chessData.xls')

evallist = []
fenlist  = []


for i in range(len(data['FEN'])):
    
    listbit = bitboard(data['FEN'][i])
    fenlist.append(listbit)
    
    
    evalbit = data['Evaluation'][i]
    try:
        evalbit = int(evalbit)
        evalbit = evalbit / 100
    
    except:
        if evalbit[1] == '+':
            evalbit = 100 + int(evalbit[1:])
        
        else:
            evalbit = -100 + int(evalbit[1:])
    
    
    evallist.append(evalbit)


evaldata = np.array(evallist)
evaldata = convert_to_tensor(evaldata)
fendata  = np.array(fenlist)
fendata  = convert_to_tensor(fendata)




# Model
def model_build(conv_size, conv_depth):
    """Builds the convolutional neural network"""
    board_matrix = layers.Input(shape=(14, 8, 8))
    
    
    x = board_matrix
    
    for _ in range(conv_depth):
        x = layers.Conv2D(filters=conv_size, 
                          kernel_size=3, 
                          padding='same', 
                          activation='relu', 
                          kernel_initializer='he_normal', 
                          use_bias=True, 
                          kernel_regularizer=regularizers.L1L2(l1=0.01, l2=0.01), 
                          dilation_rate=2)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(64, 'relu')(x)
    x = layers.Dense(1, 'sigmoid')(x)
    
    return models.Model(inputs=board_matrix, outputs=x)


model = model_build(32, 4)




# Training the model
model.compile(optimizer=optimizers.Adam(5e-4), loss="mean_squared_error")
model.summary()
history = model.fit(fendata, evaldata,
                    batch_size=2048,
                    epochs=1200,
                    verbose=1,
                    validation_split=0.1,
                    callbacks=[callbacks.ReduceLROnPlateau(monitor='loss', patience=5)])




# Saving and loading model
model.save('Anna.keras', overwrite=True, save_format='keras')
model = models.load_model("Anna.keras")