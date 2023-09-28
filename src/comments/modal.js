import React from "react";
import "./modal.css";
const Modal = ({ isOpen, closeModal, emotionData }) => {
  if (!isOpen) {
    return null;
  }

  return (
    <div className="modal-overlay" onClick={closeModal}>
      <div className="modal">
        <div className="modal-header">
          <h2>Sentiment Insights</h2>
          <button className="close-button" onClick={closeModal}>
            &times;
          </button>
        </div>
        <div className="modal-content">
          {/* Display the emotion data here */}
          <pre>{JSON.stringify(emotionData, null, 2)}</pre>
        </div>
      </div>
    </div>
  );
};

export default Modal;
