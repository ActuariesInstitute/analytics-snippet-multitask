library(keras)
library(dplyr)

df <- read.csv(url("http://www.statsci.org/data/general/motorins.txt"), sep="\t") %>%
  mutate(frequency = Claims / Insured, 
         severity = ifelse(Claims == 0, 0, Payment / Claims), 
         risk_premium = Payment / Insured)

head(df)
train_ind <- sample(seq_len(nrow(df)), size = floor(0.8 * nrow(df)))

train <- df[train_ind,]
test  <- df[-train_ind,]

summary <- df %>% summarise(
  train_average_frequency = sum(Claims) / sum(Insured),
  train_average_severity = sum(Payment) / sum(Claims),
  train_average_risk_premium = sum(Payment) / sum(Insured)
)

summary

zone_input = layer_input(shape=c(1), name='zone_input')
make_input = layer_input(shape=c(1), name='make_input')

zone_embedding = layer_embedding(output_dim=2, input_dim=7)(zone_input) %>%
  layer_reshape(target_shape=c(2))

make_embedding = layer_embedding(output_dim=2, input_dim=9)(make_input) %>% 
  layer_reshape(target_shape=c(2))

kilometres_input = layer_input(shape=c(1), name='kilometres_input')
bonus_input = layer_input(shape=c(1), name='bonus_input')

x = layer_concatenate(c(zone_embedding, make_embedding, kilometres_input, bonus_input)) %>%
  layer_dense(64, activation='relu') %>% 
  layer_dense(64, activation='relu') %>%
  layer_dense(64, activation='relu')

frequency_output    = x %>% layer_dense(1, activation='relu', name='frequency')
severity_output     = x %>% layer_dense(1, activation='relu', name='severity')
risk_premium_output = x %>% layer_dense(1, activation='relu', name='risk_premium')

model = keras_model(inputs=c(zone_input, make_input, kilometres_input, bonus_input), 
              outputs=c(frequency_output, severity_output, risk_premium_output))

model %>% compile(optimizer='adam',
              loss=list(risk_premium='mean_squared_error', 
                        frequency='poisson', 
                        severity='mean_squared_logarithmic_error'), 
              loss_weights=list(risk_premium= 1.0, frequency= 1.0, severity= 1.0))

summary(model)

InputDataTransformer <- function(x){
  list(
    kilometres_input= (x$Kilometres - 1) / 5,
    zone_input= (x$Zone - 1),
    bonus_input= (x$Bonus - 1) / 7,
    make_input= x$Make - 1
  )}

model %>% fit(
  x=InputDataTransformer(train),
  y=list(
    frequency=train$frequency / summary$train_average_frequency,
    severity=train$severity / summary$train_average_severity,
    risk_premium=train$risk_premium / summary$train_average_risk_premium),
  sample_weight=list(
    frequency=train$Insured,
    severity=train$Claims,
    risk_premium=train$Insured),
  epochs=40, batch_size=32)


predict_and_plot <- function(df, train_average_frequency, train_average_risk_premium){
  predictions <- model %>% predict(InputDataTransformer(df)) %>% as.data.frame()
  colnames(predictions) <- c("model_frequency", "model_severity", "model_risk_premium")
  
  # Reverse the normalisation
  df_new <- df %>% cbind(predictions) %>%
    mutate(model_frequency = model_frequency * train_average_frequency,
           model_risk_premium = model_risk_premium * train_average_risk_premium) %>%
    mutate(model_payment = model_risk_premium * Insured,
           model_claim_count = model_frequency * Insured)
  
  df_summary <- df_new %>%
    summarise(model_claim_count_sum = sum(model_claim_count),
              model_payment_sum = sum(model_payment))
  
  # Score, sort by lowest to higher
  df_new <- df_new %>%
    arrange(model_frequency) %>%
    mutate(model_claim_count_band = floor(cumsum(model_claim_count) / df_summary$model_claim_count_sum * 10) / 10) %>%
    arrange(model_risk_premium) %>%
    mutate(model_payment_band = floor(cumsum(model_payment) / df_summary$model_payment_sum * 10) / 10)
  
  # Summarise and plot frequency by weighted decile rank
  df_new %>% group_by(model_claim_count_band) %>%
    summarise(model_frequency = sum(model_claim_count) / sum(Insured),
              actual_frequency = sum(Claims) / sum(Insured)) %>%
    select(-model_claim_count_band) %>% as.matrix() %>% t() %>%
    barplot(xlab = "Frequency Decile", ylab="Frequency", beside = TRUE)

  # Summarise and plot risk premium by weighted decile rank
  df_new %>% group_by(model_payment_band) %>%
    summarise(model_risk_premium = sum(model_payment) / sum(Insured),
              actual_risk_premium = sum(Payment) / sum(Insured)) %>%
    select(-model_payment_band) %>% as.matrix() %>% t() %>%
    barplot(xlab = "Risk Premium Decile", ylab="Risk Premium", beside = TRUE)

  df_new
}

train = predict_and_plot(train, summary$train_average_frequency, summary$train_average_risk_premium)
test = predict_and_plot(test, summary$train_average_frequency, summary$train_average_risk_premium)

library(ggplot2)

# Change the colors manually
p <- ggplot(data=df_new, aes(x=model_payment_band, y=len, fill=supp)) +
  geom_bar(stat="identity", color="black", position=position_dodge())+
  theme_minimal() + scale_fill_brewer(palette="Blues")



# Use custom colors
p + scale_fill_manual(values=c('#999999','#E69F00'))
# Use brewer color palettes
p + scale_fill_brewer(palette="Blues")
  