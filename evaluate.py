import numpy as np
import pandas as pd
import heapq

class Evaluate():
    """Evaluation for `ARERec`."""

    def get_evaluate(self, model, test_x, test_y, id2label, batch_size, topk=10, hrcutoff=3):
        """Evaluate model.

        Args:
            model: Object of the model.
            test_x: Test set.
            test_y: Actual value of the test set.
            id2label: Mapping dictionary between IDs and its own label.
            batch_size: Batch size.
            topk: Number of top-k for evaluation metrics.
            hrcutoff: Cufoff rating of HitRate.

        Returns:
            hitrates: HitRate value.
            ndcgs: NDCG value.
        """

        hitrates_list, ndcgs_list = [], []
        map_item_score_pred = {}
        map_item_score_ideal = {}

        user = test_x[0]
        item = test_x[1]

        # Predict
        prediction = model.predict(test_x, batch_size=batch_size)
        prediction = list(map(lambda x: np.argmax(x), prediction))
        prediction = list(map(lambda x: id2label[x], prediction))

        # Relabel from one-hot vector
        label = list(map(lambda x: np.argmax(x), test_y))
        label = list(map(lambda x: id2label[x], label))

        # Store user, item, and rating together in a dictionary for both ideal and prediction.
        for i in range(len(user)):
            if user[i] not in map_item_score_pred:
                map_item_score_pred[user[i]] = {}

            if user[i] not in map_item_score_ideal:
                map_item_score_ideal[user[i]] = {}

            map_item_score_pred[user[i]][item[i]] = prediction[i]
            map_item_score_ideal[user[i]][item[i]] = label[i]

        # List top-rank items for each user and store them in a dictionary.
        for u in map_item_score_ideal:

            # Skip when the user sequence is less than top-k.
            if len(map_item_score_ideal[u]) < topk:
                continue

            # Select the top-K actual items, followed by the actual ratings.
            ideal_rank_item = heapq.nlargest(topk,
                                             map_item_score_ideal[u],
                                             key=lambda x: map_item_score_ideal[u][x])
            ideal_rank_rating = list(map(lambda x: map_item_score_ideal[u][x], ideal_rank_item))

            # Select the top-K predicted items, followed by the predicted ratings.
            pred_rank_item = heapq.nlargest(topk,
                                            map_item_score_pred[u],
                                            key=lambda x: map_item_score_pred[u][x])
            pred_rank_rating = []
            for x in ideal_rank_item:
                if x in pred_rank_item:
                    position = pred_rank_item.index(x)
                    rate = map_item_score_ideal[u][ideal_rank_item[position]]
                else:
                    rate = 0
                pred_rank_rating.append(rate)

            # Calculate HitRate and NDCG for the prediction.
            hitrate = self.get_HitRate(pred_rank_item, ideal_rank_item, ideal_rank_rating, hrcutoff)
            ndcg = self.get_NDCG(pred_rank_rating, ideal_rank_rating)

            hitrates_list.append(hitrate)
            ndcgs_list.append(ndcg)

        # Average HitRate and NDCG values.
        hitrates = np.nanmean(hitrates_list)
        ndcgs = np.nanmean(ndcgs_list)

        return hitrates, ndcgs


    def get_HitRate(self, pred_item, ideal_item, ideal_rating, rating_cutoff=3):
        """Calculate HitRate.

        Args:
            pred_item: Predicted item.
            ideal_item: Ideal item.
            ideal_rating: Ideal rating.
            rating_cutoff: Cutoff rating of HitRate.

        Returns:
            hitrates: HitRate values.
        """

        hits = []
        for i in range(len(ideal_item)):
            # Only check items with ratings higher than the cutoff threshold.
            if ideal_rating[i] >= rating_cutoff:
                if ideal_item[i] in pred_item:
                    hits.append(1)  # If hit.
                else:
                    hits.append(0)  # If not hit.
        return np.nanmean(np.array(hits))


    def get_NDCG(self, pred_rank_rating, ideal_rank_rating):
        """Calculate NDCG.

        Args:
            pred_rank_rating: Ratings of predicted items.
            ideal_rank_rating: Ratings of ideal items.

        Returns:
            ndcg: NDCG value.
        """

        def get_DCG(rank_rating, index):
            """Calculate DCG.

            Args:
                rank_rating: Ratings of ranked items.
                index: Indexes of ranked items.

            Returns:
                dcg: DCG value.
            """
            dcg = np.sum((np.exp2(rank_rating) - 1) / np.log2(index + 1))
            return dcg

        index = np.array(pd.DataFrame(ideal_rank_rating).rank(ascending=0, method='min')[0].tolist())

        dcg = get_DCG(pred_rank_rating, index)
        idcg = get_DCG(ideal_rank_rating, index)
        ndcg = dcg / idcg

        return ndcg