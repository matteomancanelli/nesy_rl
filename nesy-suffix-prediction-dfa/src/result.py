from sampling import evaluate_similarity, evaluate_satisfiability


class Result:
    def __init__(self, run_folder, run_id, model_type, train_acc, test_acc, nr_epochs, training_time):
        self.run_folder = run_folder
        self.run_id = run_id
        self.model_type = model_type
        self.train_accuracy = train_acc
        self.test_accuracy = test_acc
        self.nr_epochs = nr_epochs
        self.training_time = training_time
        self.predictions = {}

    def add_predictions(self, prefix, predictions):
        self.predictions[prefix] = predictions

    def evaluate_predictions(self, train_ds, test_ds, tensor_dfa):
        results = []
        for prefix, lists in self.predictions.items():
            for strategy in ['temperature', 'greedy']:
                results.append(
                    self.calculate_metrics(prefix, strategy, lists[f'train_{strategy}'], lists[f'test_{strategy}'],
                                           train_ds, test_ds, tensor_dfa)
                )
        return results

    def calculate_metrics(self, prefix, strategy, train_pred, test_pred, train_ds, test_ds, tensor_dfa):
        train_dl, train_dl_scaled = evaluate_similarity(train_pred, train_ds)
        test_dl, test_dl_scaled = evaluate_similarity(test_pred, test_ds)
        train_sat = evaluate_satisfiability(tensor_dfa, train_pred)
        test_sat = evaluate_satisfiability(tensor_dfa, test_pred)
        return self.get_result_row(
            prefix, strategy, train_dl, train_dl_scaled, test_dl, test_dl_scaled, train_sat, test_sat
        )

    def get_result_row(self, prefix, strategy, train_dl, train_dl_scaled, test_dl, test_dl_scaled, train_sat, test_sat):
        return {
            'run_id': self.run_id,
            'prefix length': prefix,
            'model': self.model_type,
            'sampling strategy': strategy,
            'train accuracy': self.train_accuracy,
            'test accuracy': self.test_accuracy,
            'train DL': train_dl,
            'train DL scaled': train_dl_scaled,
            'test DL': test_dl,
            'test DL scaled': test_dl_scaled,
            'train sat': train_sat,
            'test sat': test_sat,
            'nr_epochs': self.nr_epochs,
            'training_time': self.training_time
        }
