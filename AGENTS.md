# PenguinChess — Agents Development Guide

## Project Overview

PenguinChess (企鹅棋) is a two-player hexagonal board game using cube coordinates (q, r, s).
The board has 60 hexes with random values (1, 2, or 3), total sum = 99.
Each player controls 3 pieces, and the goal is to be the last player with remaining pieces.

**Tech stack**: Python (Flask backend, `uv` package manager) + Vanilla JavaScript/CSS
**Entry**: `uv run main.py` starts Flask on port 8080

---

## Architecture

```
main.py              # Flask entry point (10 lines)
templates/
  index.html         # Single-page HTML shell
statics/
  main.js            # Core game logic (769 lines) — placement, movement, game loop, replay
  board.js           # Hex class + createBoard() — board rendering and hex operations
  piece.js           # Piece class — piece placement and movement
  player.js          # Player class — score and statistics
  style.css          # Board and piece styling
```

**Data flow**: `index.html` loads `main.js` as ES module → imports from `board.js`, `piece.js`, `player.js` → game initializes on DOMContentLoaded → event listeners drive game phases.

---

## Game Rules

### Board Layout
- 60 hexagonal tiles in a parallelogram arrangement
- Cube coordinates (q, r, s) where q + r + s = 0
- q ranges from -4 to 3; r range varies by row
- Each hex has a `value` (1, 2, or 3); total = 99
- Hexes with value > 0 are active; value = -1 means eliminated

### Placement Phase
1. Players alternate placing pieces, starting with Player 1 ("Milky")
2. Each player places exactly 3 pieces
3. Piece IDs: Player 1 gets 4, 6, 8 — Player 2 gets 5, 7, 9
4. Placing a piece on a hex scores that hex's value for the player
5. Both players complete placement before movement phase begins

### Movement Phase
1. Players alternate turns (Player 1 on odd turns, Player 2 on even)
2. A piece can move along any axis where q, r, or q+r matches the destination
3. Movement is blocked by:
   - Other pieces (any player's)
   - Hexes with value <= 0
   - Intermediate hexes along the path must be empty
4. Moving to a hex scores its value for the player
5. After each turn, hexes not connected to any piece (via adjacency) are eliminated (value → -1)
6. A piece with no valid moves is destroyed
7. **Game over**: When one player has no pieces left → surviving player collects all remaining hex values

### Win Condition
Last player with pieces on the board wins. Final score = points collected during placement + movement + remaining hex values after opponent's elimination.

---

## Code Structure

### Global State (main.js)
```javascript
hexes = []      // Array of Hex instances on the board
pieces = []     // Array of Piece instances currently on the board
players = []    // [Player1, Player2]
history = []    // Game record for replay/export/import
totalValue = 99 // Sum of all hex values
```

### Key Classes

**Hex (board.js)**
- Properties: q, r, s, value, centerX, centerY, left, top, element
- Methods: `getAdjacentCoords()`, `getConnectedHexes(allHexes)`, `updateStatus(value)`, `removeClickHandler()`
- Coordinate system: cube coordinates; adjustment applied to r for board layout (qAdjustments map)

**Piece (piece.js)**
- Properties: id, hex (current hex), element
- Methods: `placeToHex(hex)`, `moveToHex(newHex)`, `destroySelf()`
- ID parity: even = Player 1, odd = Player 2

**Player (player.js)**
- Properties: id, name, score, gamesWon, gamesLost, gamesDrawn, isComputer
- Methods: `addScore(points)`, `recordWin()`, `recordLoss()`, `recordDraw()`, `reset()`

### Game Loop (main.js)
```
initializeGame()
  → generateSequence(99)         # Random hex values
  → createBoard({valueSequence}) # Render 60 hexes
  → placePieces()               # Placement phase (await user input)
  → while (!gameover) turn(i++) # Movement phase loop
```

### Movement Logic (`calculatePossibleMoves`, main.js)
1. Filter hexes where `q` OR `r` OR `q+r` matches current hex
2. Exclude occupied hexes and hexes with value <= 0
3. For each candidate, verify all intermediate hexes along the path are clear
4. Return valid destination array

### Hex Connectivity (`checkAndRemoveHexes`, main.js)
1. From each piece's hex, recursively add all adjacent hexes (value > 0) to connected set
2. Any hex not in the connected set gets `updateStatus(-1)` (eliminated)

---

## Features

### Replay System
- **Export**: `history` array serialized to JSON → downloaded as `game-history.json`
- **Import**: File input reads JSON → `replayHistory(history)` replays all actions with 1s delay between steps
- **Toggle coords**: Button switches hex labels between value display and cube coordinate display

### History Record Format
```javascript
// Initialization
{ type: "initialize", valueSequence: [...], players: [...] }

// Piece placement
{ type: "place", pieceId: 4, hex: {q, r, s}, player: Player }

// Piece movement
{ type: "move", pieceId: 4, fromHex: {q,r,s}, toHex: {q,r,s}, player: Player }
```

---

## Known Issues / Caveats

- `gameovercheck()` uses `piece.id % 2` to determine ownership but the actual ownership is based on integer division (`Math.floor(piece.id / 2)`) — bug present in current code
- `gameovercheck()` has dead code after `throw error` (unreachable `return true`)
- `replayHistory()` creates new hexes array locally but global `hexes` is not updated, which may cause issues with subsequent operations
- `aftergame()` is empty — no end-game settlement logic beyond score accumulation
- CSS hardcodes board size at 600x600px; not responsive
- No input validation on imported JSON files

---

## Running Locally

```bash
cd /mnt/e/programming/penguinchess
uv run main.py        # Starts Flask on http://localhost:8080
# or
uv run -- python main.py
```

Dependencies declared in `pyproject.toml` (currently none beyond Flask implicit).

---

## Development Notes

- The Flask static server serves everything under `statics/` at `../statics/` from `templates/index.html` — paths rely on relative traversal
- ES module `type="module"` is used in `index.html` for `main.js`
- No build step, no bundler — pure browser-native ES modules
- Board rendering uses absolute positioning on a 600x600 container; hex positions calculated from cube-to-pixel projection
