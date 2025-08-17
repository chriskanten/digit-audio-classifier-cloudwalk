import torch
from src.data_loader import load_digit_dataset
from src.model import AudioDigitClassifier  
from src.trainer import ModelTrainer

def main():
    # Load the dataset
    train_loader, test_loader = load_digit_dataset()

    # Check if CUDA is available and set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    # Initialize the model
    model = AudioDigitClassifier(input_dim=13, num_classes=10, hidden_dim=64)
    
    # Initialize the trainer
    trainer = ModelTrainer(model)

    # Train the model
    num_epochs = 50
    best_accuracy = 0
    print("Starting training...")

    for epoch in range(num_epochs):
        train_loss, train_accuracy = trainer.train(train_loader)
        print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

        if train_accuracy > best_accuracy:
            best_accuracy = train_accuracy
            model.save(f"best_model_epoch_{epoch+1}.pth")
            print(f"Best model saved at epoch {epoch+1} with accuracy: {best_accuracy:.2f}%")   

    print("Training complete.")

    # live prediction
    print("Starting live prediction...")
    from src.live_predictor import LivePredictor
    live_predictor = LivePredictor(model, device)
    
    while True:
        try:
            predicted_digit, confidence = live_predictor.predict()
            print(f"Predicted Digit: {predicted_digit}, Confidence: {confidence:.2f}")
        except KeyboardInterrupt:
            print("Live prediction stopped.")
            break

if __name__ == "__main__":
    main()
