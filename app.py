import streamlit as st
import chess
import chess.svg
import numpy as np
import tensorflow as tf
import pandas as pd

# Load the trained model
model = tf.keras.models.load_model('best_model_3d.h5')

# Preprocessing Functions
castlingdict = {'K': 'CA_K', 'Q': 'CA_Q', 'k': 'CA_k', 'q': 'CA_q'}
tonum3d = {
    'r': -np.array([0, 8, 0, 8, 0, 8, 0, 8], dtype=np.float32),
    'n': -np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5], dtype=np.float32),
    'b': -np.array([8, 0, 8, 0, 8, 0, 8, 0], dtype=np.float32),
    'q': -np.array([8, 8, 8, 8, 8, 8, 8, 8], dtype=np.float32),
    'k': -np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
    'p': -np.array([0, 0, 0, 0, 0, 1, 0, 0], dtype=np.float32),

    'R': np.array([0, 8, 0, 8, 0, 8, 0, 8], dtype=np.float32),
    'N': np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5], dtype=np.float32),
    'B': np.array([8, 0, 8, 0, 8, 0, 8, 0], dtype=np.float32),
    'Q': np.array([8, 8, 8, 8, 8, 8, 8, 8], dtype=np.float32),
    'K': np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
    'P': np.array([0, 0, 0, 0, 0, 1, 0, 0], dtype=np.float32),
}

white_weights ={'P': 0.55766841775115, 'R': -0.22384384406494998, 'N': -0.6152549360423548, 
                'B': -0.6641308111861762, 'Q': -1.9615679236600596, 'K': 0.0}
black_weights ={'p': 0.04473323407809408, 'r': 0.07230358408945219, 
                'n': -0.057320978614105775, 'b': 0.13213246344153942, 
                'q': 0.7980719034607346, 'k': -6.661338147750939e-16}

def fen2cube(fen, tonum3d):
    board = np.zeros((8, 8, 8), dtype=np.float32)
    for i, row in enumerate(fen.split('/')):
        j = 0
        for c in row:
            if c.isdigit():
                j += int(c)
            else:
                board[i, j] = tonum3d[c]
                j += 1
    return board

def numbw(fen_str):
    blk = {'p': 0, 'r': 0, 'n': 0, 'b': 0, 'q': 0, 'k': 0}
    wht = {'P': 0, 'R': 0, 'N': 0, 'B': 0, 'Q': 0, 'K': 0}
    for c in fen_str:
        if c.islower():
            blk[c] = blk.get(c, 0) + 1
        elif c.isupper():
            wht[c] = wht.get(c, 0) + 1
    return blk, wht

def weighted_sum(piece_counts, piece_weights):
    weighted_sum = 0
    for piece, count in piece_counts.items():
        weight = piece_weights.get(piece, 0)
        weighted_sum += count * weight
    return weighted_sum

# Streamlit UI
st.set_page_config(layout="wide")
st.title("Chess Board Viewer with Prediction")

fen_input = st.text_input("FEN string", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

try:
    board = chess.Board(fen_input)
    st.success("Valid FEN string!")
except Exception as e:
    st.error("Invalid FEN string: " + str(e))
    st.stop()

svg_board = chess.svg.board(board=board, size=400)
col1, col2 = st.columns([3, 2])

with col1:
    st.components.v1.html(svg_board, height=450, scrolling=False)

if st.button("Predict Winning Probability"):
    fen_cols = fen_input.split()
    fen_board, next_move, CA, _, hm, fm = fen_cols
    
    next_move = -1 if next_move == 'b' else 1
    CA_K = int('K' in CA)
    CA_Q = int('Q' in CA)
    CA_k = int('k' in CA)
    CA_q = int('q' in CA)
    hm = int(hm)
    fm = int(fm)
    
    cube1 = fen2cube(fen_board, tonum3d)
    num_blk, num_wht = numbw(fen_board)
    wtdb = weighted_sum(num_blk, black_weights)
    wtdw = weighted_sum(num_wht, white_weights)
    
    vecinp1 = np.array([[CA_K, CA_Q, CA_k, CA_q]], dtype=np.float32)
    vecinp2 = np.array([[hm, fm, next_move]], dtype=np.float32)
    vecinp3 = np.array([[wtdb, wtdw]], dtype=np.float32)
    
    cube1 = np.expand_dims(cube1, axis=0)
    
    input_data = [cube1, vecinp1, vecinp2, vecinp3]
    
    # Prediction
    prediction = model.predict(input_data)
    prediction = prediction[0][0]
    
    st.write(f"Predicted winning probability: {prediction:.2f}")
    
    slider_value = st.slider(
        "Winning Probability",
        min_value=-1.0,
        max_value=1.0,
        value=float(prediction),
        step=0.01,
        format="%.2f"
    )
    
    st.markdown("""
        <style>
            .slider-container {
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .black-square, .white-square {
                width: 12px;
                height: 12px;
                border-radius: 3px;
            }
            .black-square {
                background-color: black;
                border: 1px solid gray;
            }
                
            .white-square {
                background-color: white;
                border: 1px solid gray;
            }
        </style>
        <div class="slider-container">
            <div class="black-square"></div>
            <div class="white-square"></div>
        </div>
    """, unsafe_allow_html=True)
