from classifier import train
if __name__ == "__main__":
    print("🚀 Starting training...")
    train("data/complaints.csv")
    print("\n✅ Training complete! Model saved to models/model.joblib")