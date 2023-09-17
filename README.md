# BlackJackVision

This project aims to detect and identify playing cards on a table using computer vision techniques. It is developed as a part of the CS 5330 - Pattern Recognition and Computer Vision course.

## Team Members:
- Aparna Krishnan
- Farhan Sarfraz
- Suparna Srinivasan


## Features:

1. **Card Detection** - Uses computer vision techniques to detect playing cards on a table.
2. **Card Identification** - Identifies the rank and suit of detected playing cards.
3. **Blackjack Logic** - Can be extended to integrate with the logic of the Blackjack game.

## Dependencies:

- OpenCV
- Numpy
- tkinter

## Classes:

1. **Query_card** - Represents a playing card. Stores its contour, dimensions, images of its rank and suit, etc.
2. **Train_ranks** - Represents a rank used for training the card detection system. Contains an image of the rank and its name.
3. **Train_suits** - Represents a suit used for training the card detection system. Contains an image of the suit and its name.

## Functions:

1. `ranks(filepath)`: Loads rank images for training.
2. `suits(filepath)`: Loads suit images for training.
3. `preprocess(image)`: Processes the input image to detect cards.
4. `find_cards(thresh_image)`: Identifies contours in the image that could be cards.
5. `preprocess_card(contour, image)`: Further processes an identified card to extract its features.
6. `match(qCard, train_ranks, train_suits)`: Matches a card's features with trained data to identify its rank and suit.
7. `store_result(n, cards)`: Stores the result of the card identification for the game logic.
8. `draw(image, qCard)`: Draws the identified card's rank and suit on the image.
9. `flattener(image, pts, w, h)`: Flattens a card's image for easier processing.

## Quickstart:

1. **Setup Virtual Environment** (Recommended)
```bash
python3 -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Application**
```bash
python main.py
```

## Notes:

- Ensure the training images for ranks and suits are placed in the appropriate directory.
- Adjust the constants in the code if the card detection is not accurate for your setup.
- Remember to integrate the card detection logic with the Blackjack game rules for a complete game experience.
