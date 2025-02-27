from implicit.als import AlternatingLeastSquares

def train_model(train_matrix):
    model = AlternatingLeastSquares(
        factors=100,
        regularization=0.1,
        iterations=15,
        random_state=42
    )
    model.fit(train_matrix)
    return model