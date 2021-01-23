This is my attempt at coding KNN classifier from scratch. 
Some part of it, specifically the namedtuple segment, is hard to reconcile with typical ML data. 
Therefore, I just used sklearn for the practical example. 
The gist of the algorithm is basically this:
def knn_classify(k: int,
                 labeled_points: List[LabeledPoint],
                 new_point: Vector) -> str:

    # Order the labeled points from nearest to farthest.
    by_distance = sorted(labeled_points,
                         key=lambda lp: distance(lp.point, new_point))

    # Find the labels for the k closest
    k_nearest_labels = [lp.label for lp in by_distance[:k]]

    # and let them vote.
    return majority_vote(k_nearest_labels)

NamedTuple helps with keeping labels and points "tied" together in the sorting process. 
