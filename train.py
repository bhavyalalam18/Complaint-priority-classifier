from classifier import train

if __name__ == "__main__":
    train(
        csv_path="data/complaints.csv",
        model_path="models/model.joblib"
    )