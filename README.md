# Laurelia: All-Purpose xLSTM Chatbot

[Versión en Español abajo](#español)

Laurelia is a global, all-purpose project based on the **xLSTM (Extended Long Short-Term Memory)** architecture. Moving away from its origins in financial forecasting, Laurelia now functions as a powerful GPT-like chatbot, providing a high-performance implementation for general use.

The project has been fully ported from **Python** and the **Burn** framework to **Candle** (by Hugging Face), aiming to strictly follow the architectural guidelines described in the official xLSTM research paper: ["xLSTM: Extended Long Short-Term Memory"](https://arxiv.org/abs/2405.04517).

## Key Features

*   **xLSTM Architecture**: High-fidelity implementation of both **sLSTM** (scalar LSTM with exponential gating) and **mLSTM** (matrix LSTM with covariance memory).
*   **Chatbot Interface**: Designed for conversational tasks and general natural language processing.
*   **Performance with Candle**: Optimized using the Candle deep learning framework in Rust for fast inference and training.
*   **Scientific Compliance**: The implementation is built to respect the official xLSTM paper specifications as closely as possible.

## Getting Started

1.  **Preparation**: Ensure you have the `xlstm_chat_model.safetensors` and `tokenizer.json` files in the root directory.
2.  **Run Chat**: 
    ```bash
    cargo run --release --bin xlstmchat
    ```
    Other variants are available: `mlstmchat` and `slstmchat`.

---

<a name="español"></a>

# Laurelia: Chatbot xLSTM de Uso Global

Laurelia es un proyecto de uso global y propósito general basado en la arquitectura **xLSTM (Extended Long Short-Term Memory)**. Dejando atrás sus inicios en la predicción financiera, Laurelia ahora funciona como un potente chatbot estilo GPT, ofreciendo una implementación de alto rendimiento para uso general.

El proyecto ha sido portado completamente de **Python** y el framework **Burn** a **Candle** (de Hugging Face), con el objetivo de seguir estrictamente las directrices arquitectónicas descritas en el paper oficial de investigación de xLSTM: ["xLSTM: Extended Long Short-Term Memory"](https://arxiv.org/abs/2405.04517).

## Características Principales

*   **Arquitectura xLSTM**: Implementación fiel de **sLSTM** (LSTM escalar con compuertas exponenciales) y **mLSTM** (LSTM matricial con memoria de covarianza).
*   **Interfaz de Chatbot**: Diseñado para tareas conversacionales y procesamiento de lenguaje natural general.
*   **Rendimiento con Candle**: Optimizado utilizando el framework Candle en Rust para una inferencia y entrenamiento rápidos.
*   **Cumplimiento Científico**: La implementación está construida para respetar las especificaciones del paper oficial de xLSTM lo más fielmente posible.

## Cómo empezar

1.  **Preparación**: Asegúrate de tener los archivos `xlstm_chat_model.safetensors` y `tokenizer.json` en el directorio raíz.
2.  **Iniciar Chat**:
    ```bash
    cargo run --release --bin xlstmchat
    ```
    También están disponibles las variantes: `mlstmchat` y `slstmchat`.

---

## License / Licencia

**English**: This project is licensed under the **MIT License** and **Apache License 2.0**.

**Español**: Este proyecto está bajo la licencia **MIT** y **Apache 2.0**.
