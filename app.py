# app.py
# Connect 4 Game with AI using Minimax Algorithm + Flask Web App

from flask import Flask, render_template_string, request, jsonify
import numpy as np
import math
import random

app = Flask(__name__)

ROWS = 6
COLS = 7

PLAYER = 1
AI = 2

EMPTY = 0
WINDOW_LENGTH = 4

# ----------------------------
# GAME LOGIC
# ----------------------------

def create_board():
    return np.zeros((ROWS, COLS), dtype=int)

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_valid_location(board, col):
    return board[ROWS - 1][col] == 0

def get_next_open_row(board, col):
    for r in range(ROWS):
        if board[r][col] == 0:
            return r

def print_board(board):
    print(np.flip(board, 0))

def winning_move(board, piece):

    # Horizontal
    for c in range(COLS - 3):
        for r in range(ROWS):
            if (
                board[r][c] == piece and
                board[r][c + 1] == piece and
                board[r][c + 2] == piece and
                board[r][c + 3] == piece
            ):
                return True

    # Vertical
    for c in range(COLS):
        for r in range(ROWS - 3):
            if (
                board[r][c] == piece and
                board[r + 1][c] == piece and
                board[r + 2][c] == piece and
                board[r + 3][c] == piece
            ):
                return True

    # Positive diagonal
    for c in range(COLS - 3):
        for r in range(ROWS - 3):
            if (
                board[r][c] == piece and
                board[r + 1][c + 1] == piece and
                board[r + 2][c + 2] == piece and
                board[r + 3][c + 3] == piece
            ):
                return True

    # Negative diagonal
    for c in range(COLS - 3):
        for r in range(3, ROWS):
            if (
                board[r][c] == piece and
                board[r - 1][c + 1] == piece and
                board[r - 2][c + 2] == piece and
                board[r - 3][c + 3] == piece
            ):
                return True

    return False

# ----------------------------
# AI SECTION - MINIMAX
# ----------------------------

def evaluate_window(window, piece):

    score = 0
    opp_piece = PLAYER

    if piece == PLAYER:
        opp_piece = AI

    if window.count(piece) == 4:
        score += 100

    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5

    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2

    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 4

    return score

def score_position(board, piece):

    score = 0

    # Center column preference
    center_array = [int(i) for i in list(board[:, COLS // 2])]
    center_count = center_array.count(piece)
    score += center_count * 3

    # Horizontal
    for r in range(ROWS):
        row_array = [int(i) for i in list(board[r, :])]

        for c in range(COLS - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Vertical
    for c in range(COLS):
        col_array = [int(i) for i in list(board[:, c])]

        for r in range(ROWS - 3):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Positive diagonal
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    # Negative diagonal
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [board[r + 3 - i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score

def get_valid_locations(board):

    valid_locations = []

    for col in range(COLS):
        if is_valid_location(board, col):
            valid_locations.append(col)

    return valid_locations

def is_terminal_node(board):

    return (
        winning_move(board, PLAYER) or
        winning_move(board, AI) or
        len(get_valid_locations(board)) == 0
    )

def minimax(board, depth, alpha, beta, maximizingPlayer):

    valid_locations = get_valid_locations(board)
    terminal = is_terminal_node(board)

    if depth == 0 or terminal:

        if terminal:

            if winning_move(board, AI):
                return (None, 100000000000)

            elif winning_move(board, PLAYER):
                return (None, -100000000000)

            else:
                return (None, 0)

        else:
            return (None, score_position(board, AI))

    if maximizingPlayer:

        value = -math.inf
        column = random.choice(valid_locations)

        for col in valid_locations:

            row = get_next_open_row(board, col)
            b_copy = board.copy()

            drop_piece(b_copy, row, col, AI)

            new_score = minimax(
                b_copy,
                depth - 1,
                alpha,
                beta,
                False
            )[1]

            if new_score > value:
                value = new_score
                column = col

            alpha = max(alpha, value)

            if alpha >= beta:
                break

        return column, value

    else:

        value = math.inf
        column = random.choice(valid_locations)

        for col in valid_locations:

            row = get_next_open_row(board, col)
            b_copy = board.copy()

            drop_piece(b_copy, row, col, PLAYER)

            new_score = minimax(
                b_copy,
                depth - 1,
                alpha,
                beta,
                True
            )[1]

            if new_score < value:
                value = new_score
                column = col

            beta = min(beta, value)

            if alpha >= beta:
                break

        return column, value

# ----------------------------
# FLASK ROUTES
# ----------------------------

board = create_board()

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Connect 4 AI</title>

    <style>

        body{
            font-family: Arial;
            text-align:center;
            background:#1e1e2f;
            color:white;
        }

        table{
            margin:auto;
            border-spacing:10px;
            background:#004aad;
            padding:10px;
            border-radius:20px;
        }

        td{
            width:70px;
            height:70px;
            border-radius:50%;
            background:white;
            cursor:pointer;
        }

        .player{
            background:red;
        }

        .ai{
            background:yellow;
        }

    </style>
</head>

<body>

<h1>Connect 4 vs AI</h1>

<table id="board"></table>

<h2 id="status"></h2>

<script>

let board = Array(6).fill().map(() => Array(7).fill(0));

function drawBoard(){

    let table = document.getElementById("board");
    table.innerHTML = "";

    for(let r=5; r>=0; r--){

        let row = document.createElement("tr");

        for(let c=0; c<7; c++){

            let cell = document.createElement("td");

            if(board[r][c] == 1){
                cell.classList.add("player");
            }

            if(board[r][c] == 2){
                cell.classList.add("ai");
            }

            cell.onclick = () => makeMove(c);

            row.appendChild(cell);
        }

        table.appendChild(row);
    }
}

async function makeMove(col){

    let response = await fetch("/move", {
        method:"POST",
        headers:{
            "Content-Type":"application/json"
        },
        body:JSON.stringify({col:col})
    });

    let data = await response.json();

    board = data.board;

    drawBoard();

    document.getElementById("status").innerText = data.message;
}

drawBoard();

</script>

</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/move", methods=["POST"])
def move():

    global board

    data = request.get_json()
    col = data["col"]

    message = ""

    # Player move
    if is_valid_location(board, col):

        row = get_next_open_row(board, col)
        drop_piece(board, row, col, PLAYER)

        if winning_move(board, PLAYER):

            message = "You Win!"

            response = {
                "board": board.tolist(),
                "message": message
            }

            board = create_board()
            return jsonify(response)

        # AI move
        ai_col, minimax_score = minimax(
            board,
            4,
            -math.inf,
            math.inf,
            True
        )

        if is_valid_location(board, ai_col):

            ai_row = get_next_open_row(board, ai_col)
            drop_piece(board, ai_row, ai_col, AI)

            if winning_move(board, AI):

                message = "AI Wins!"

                response = {
                    "board": board.tolist(),
                    "message": message
                }

                board = create_board()
                return jsonify(response)

    return jsonify({
        "board": board.tolist(),
        "message": message
    })

# ----------------------------
# RUN
# ----------------------------

if __name__ == "__main__":
    app.run(debug=True)