from sklearn.neighbors import KNeighborsClassifier
KN = KNeighborsClassifier(n_neighbors=200)
KN.fit(train_X, train_y)