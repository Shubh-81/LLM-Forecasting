# Air Quality Forecasting using Large Language Models

**Author:** Shubh Agarwal - 21110205, Ishva Patel - 21110082

## Problem Definition

The project aims to utilize large language models for air quality forecasting, specifically leveraging the next-token prediction capabilities of Large Language Models for time series forecasting.

## Preliminary Literature

1. **Large Language Models Are Zero-Shot Time Series Forecasters Paper by NYU:** The paper discusses the use of large language models for time series forecasting, specifically focusing on next token prediction for time series forecasting.
2. **Zero-Shot Time Series Forecasting using Transformers - Moment MOMENT: A Family of Open Time-Series Foundation Models (CMU):** The paper discusses the use of transformers for time series forecasting, specifically focusing on the Moment MOMENT framework.
3. **Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting:** The paper presents Lag-Llama, a foundation model for univariate probabilistic time series forecasting based on a decoder-only transformer architecture that uses lags as covariates.

## Code Structure

The code is structured as follows:

- `zero_shot.ipynb`: Contains the implementation of Zero Shot Time Series Forecasting in which we can plug in any model from HuggingFace from choice. 

- `few-shot.ipynb`: Contains the implementation of Few Shot learning using Langchain on any model from HuggingFace. 

- `few-shot-outlines.ipynb`: Contains the implementation of Few Shot learning using structured output using Outlines library on any model from HuggingFace.

The inputs of the LLMTime Class are as follows:

- `train`: The training data for the time series forecasting.
- `test`: The test data for the time series forecasting.
- `alpha`: A parameter used in the get_scaler method. Default value is 0.95.
- `beta`: A parameter used in the get_scaler method. Default value is 0.3.
- `batch_length`: The length of the batch used in few-shot learning. Default value is 400.
- `basic`: A boolean parameter used in the get_scaler method. Default value is False.
- `temperature`: The temperature parameter used in the HuggingFace model. Default value is 0.5.
- `do_sample`: A boolean parameter used in the HuggingFace model. Default value is True.
- `model_name`: The name of the HuggingFace model to be used. Default value is "mistralai/Mistral-7B-v0.1".
- `repetition_penalty`: The repetition penalty parameter used in the HuggingFace model. Default value is 1.0.
- `load_in_4bit`: Option to use quantizied models. Default value is True.
- `num_samples`: Number of samples to be taken. Defaults to 10.

## Results

Some of the results obtained are as follows:

Sine + Cosine Function:
![Sine + Cosine Curve](zero_shot_results/Sine+Cosine/mistral-7b-v0.1/Mistral-7B-v0.1-zero-shot-2024-04-06_14-10-16-831.png)

Gas Rate CO2 Dataset (Multivariate):
![Gas Rate CO2](zero_shot_results/GreenHouseCO2/gemma-2b/gemma-2b-zero-shot-2024-03-24_13-15-53-631.png)

These were some promising results, but further analysis is required to validate the performance of the models.
Further plots can be found in the `zero_shot_results` and `few_shot_results` directory. Baseline results can be found in the `lstm_results` and `arima_results` directory.

Datasets used for testing can be found in `datasets.ipynb`. Furthe code for obtaining baseline LSTM results can be found in `lstm.ipynb`.

## References

1. "Large Language Models Are Zero-Shot Time Series Forecasters Paper by NYU" [Link](https://arxiv.org/abs/2310.07820)
2. "Zero-Shot Time Series Forecasting using Transformers - Moment MOMENT: A Family of Open Time-Series Foundation Models (CMU)" [Link](https://arxiv.org/abs/2402.03885)
3. "Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting" [Link](https://arxiv.org/html/2310.08278v3#:~:text=By%20applying%20a%20decomposition%20technique,%2C%202005%3B%20Hyndman%20%26%20Athanasopoulos%2C)
