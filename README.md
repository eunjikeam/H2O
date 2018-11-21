Machine Learning with R and H2O
=============

<br>

1. h2o Setting, 환경 확인
-------------

```
library(h2o)

# H2O 클라우드 만들기
h2o.init(nthreads = -1)  # -1 : 모든 사용가능한 스레드 사용
h2o.removeAll()          # 현재 돌아가고 있는 클러스터가 있다면 모두 지움
```

* cluster의 이름, 버전, 사용 가능한 메모리, 환경 등에 대한 정보가 출력된다.

<br>

2. Demo : GLM
-------------

아래와 같은 작업 실습  
    * import a file  
    * Define significant data  
    * View data  
    * Create testing and training sets using sampling  

```
airlinesURL = "https://s3.amazonaws.com/h2o-airlines-unpacked/allyears2k.csv"
# file 불러오기
airlines.hex = h2o.importFile(path = airlinesURL, destination_frame = "airlines.hex")
# destination_file : h2o cluster 내에서의 데이터 명 지정

summary(airlines.hex)
str(airlines.hex)
head(airlines.hex)

quantile(x = airlines.hex$ArrDelay, na.rm = T)   # 4분위수 확인
h2o.hist(airlines.hex$ArrDelay)                  # 히스토그램 출력

# airport별로 비행 횟수 출력
originFlights <- h2o.group_by(data = airlines.hex, by = "Origin", nrow("Origin")
                             , gb.control = list(na.methods = "rm"))
originFlights.R <- as.data.frame(originFlights)

head(originFlights.R)

# 월별 비행 횟수 출력
flightsByMonth <- h2o.group_by(data = airlines.hex, by = "Month", nrow("Month"),
                              gb.control = list(na.methods = "rm"))
flightsByMonth.R <- as.data.frame(flightsByMonth)

head(flightsByMonth.R)

# 월별 비행 취소 횟수 출력
which(colnames(airlines.hex) == "Cancelled")
cancellationsByMonth <- h2o.group_by(data = airlines.hex, by = "Month", sum("Cancelled"),
                                     gb.control = list(na.methods = "rm"))
cancellationsByMonth.R <- as.data.frame(cancellationsByMonth)
head(cancellationsByMonth.R)

# 월별 비행 취소율
cancellation_rate = cancellationsByMonth$sum_Cancelled/flightsByMonth$nrow
cancellation_rate

rates_table = h2o.cbind(flightsByMonth$Month, cancellation_rate)
rates_table.R = as.data.frame(rates_table)

# training 과 test set으로 나누기 위해 데이터 쪼개기
airlines.split = h2o.splitFrame(data = airlines.hex, ratios = 0.85)

airlines.train <- airlines.split[[1]]
airlines.test <- airlines.split[[2]]

h2o.table(airlines.train$Cancelled)
h2o.table(airlines.test$Cancelled)

# GLM 시작
Y <- "IsDepDelayed"
X <- c("Origin", "Dest", "DayofMonth", "Year", "UniqueCarrier", "DayOfWeek", "Month", 
       "DepTime","ArrTime", "Distance")

airlines.glm <- h2o.glm(training_frame = airlines.train, x = X, y = Y, 
                        family = "binomial", alpha = 0.5)
# family : 종속변수의 분포함수

summary(airlines.glm)

# Predict
pred <- h2o.predict(object = airlines.glm, newdata = airlines.test)
summary(pred)

```

<br>

3. Data Manipulation in R
-------------

### 3.1 Importing & uploading Files

`h2o.importFile` function : h2o.importFile(path = path, destination_frame = name)  
    * `path` : data 경로  
    * `destination_frame` : R에서 사용될 데이터 이름  
    
```
irisPath <- system.file("extdata", "iris.csv", package = "h2o")  # irisfile의 경로 지정
iris.hex <- h2o.importFile(path = irisPath, destination_frame = "iris.hex") # 데이터 불러오기
summary(iris.hex)
```

* 데이터 경로는 IP주소로 HDFS 에서 불러오는 것도 가능하다.

<br>

H2O cluster에 파일 업로드  
`h2o.uploadFile` function : h2o.uploadFile(path = path, destination_frame = name)  
    * `path` : 내보낼 파일 경로  
    * `destination_frame` : 내보낼 파일이름.확장자  

```
irisPath <- system.file("extdata", "iris.csv", package = "h2o")  # irisfile의 경로 지정
iris.hex <- h2o.uploadFile(path = irisPath, destination_frame = "iris.hex") # 데이터 업로드
```

<br>

### 3.2 Finding & Converting Factors

범주형인 변수를 찾거나 변환하기  

`h2o.anyFactor` function : h2o.anyfunction(data)  
    * data 안에 factor인 변수가 있는지 확인하는 함수  
    
`as.factor` function : as.factor(vector)  
    * 벡터인 변수를 factor형으로 바꾸는 함수  
    
```
irisPath <- system.file("extdata", "iris_wheader.csv", package = "h2o")
iris.hex <- h2o.importFile(path = irisPath)
h2o.anyFactor(iris.hex)              # 데이터 안에 factor형 변수가 있는지 확인
str(iris.hex)

prosPath <- system.file("extdata", "prostate.csv", package = "h2o")
prostate.hex <- h2o.importFile(path = prosPath)

as.factor(prostate.hex[,4])          # 변수 factor로 만들기

prostate.hex[,4] <- as.factor(prostate.hex[,4])
summary(prostate.hex[,4])            # 변수가 factor가 되어서 factor별로 counting 되었다.
```

<br>

### 3.3 Converting & Transferring Data Frames

Data Frame을 가공하거나 변환하기  
`as.h2o` function : as.h2o(data, destination_frame = name.hex)  
    * `destination_frame` : h2o에 맞는 데이터 프레임 hex로 만든다.  
``` 
prosPath <- system.file("extdata", "prostate.csv", package = "h2o")
prostate.hex <- h2o.importFile(path = prosPath, destination_frame = "prostate.hex")

prostate.R <- as.data.frame(prostate.hex)    # hex data data frame으로 변환
summary(prostate.R)

iris.hex <- as.h2o(iris, destination_frame = "iris.hex") # data frame을 h2o frame으로 변환
```

* `extdata`는 패키지 내부에 있는 데이터를 말한다. 별다른 경로가 필요하지 않으며 뒤에 `package = `만 쓰면 된다.

<br>

### 3.4 summarizig data & table 

```
h2o.ls()                                    # h2o 상에 올라와 있는 objet들의 list를 출력해준다.

h2o.table(prostate.hex[,c("AGE","RACE")])   # data의 변수에 따른 count를 보여줌
```

<br>

### 3.5 Generating Random Numbers

훈련데이터랑 테스트 데이터를 나누는 방법은 두가지가 있다.

`h2o.runif` function : h2o.runif(data)  
    * 데이터에서 uniform 분포에 따라 random number 추출  

`h2o.splitFrame` function : h2o.splitFrame(data, ratios = rate)  
    * `data` : 쪼갤 data 입력  
    * `ratios` : 얼만큼의 비율로 쪼갤건지 설정. 벡터를 이용해 여러개로 쪼갤 수 도 있음  

```
### 1. h2o.runif()를 이용한 training set, test set 만들기
s <- h2o.runif(prostate.hex)          # prostate 데이터에서 uniform 분포의 random 변수 생성.
summary(s)

prostate.train <- prostate.hex[s <= 0.8,]                      # 80%의 training set 생성
prostate.train <- h2o.assign(prostate.train, "prostate.train") # cluster에 assign
prostate.test <- prostate.hex[s > 0.8,]                        # 20%의 test set 생성
prostate.test <- h2o.assign(prostate.test, "prostate.test")    # cluster에 assign

nrow(prostate.train) + nrow(prostate.test)                     # 데이터의 행의 개수 확인
nrow(prostate.hex)

### 2. h2o.splitFrame()를 이용한 training set, test set 만들기
prostate.split <- h2o.splitFrame(data = prostate.hex, ratios = 0.75) # 0.75를 기준으로 데이터를 분리함.
prostate.train <- prostate.split[[1]]                          # training set
prostate.test <- prostate.split[[2]]                           # test set
```

<br>

### 3.6 Getting Frames & Models

`h2o.getFrame()` : h2o cluster에 있는 data frame 불러오기  
`h2o.getModel()` : h2o cluster에 있는 Model 불러오기  

```
# prostate.hex <- h2o.getFrame(id = "prostate.hex_sid_85ce_21") # h2o cluster에서 데이터 불러오기
# prostate.hex

# gbm.model <- h2o.getModel(model_id = "GBM_8e4591a9b413407b983d73fbd9eb44cf")
# model 불러오기

h2o.rm("prostate.train")              # cluster에 올라와 있는 데이터 지우기
h2o.ls()
h2o.removeAll()                      # cluster에 있는 모든 데이터 제거

```

<br>

4. Running Models
-------------

### 4.1 Gradient Boosting Machine (GBM)

GBM은 앙상블 러닝에서 모형을 향상시켜주는데 사용된다.   
`h2o.gbm` function : h2o.gbm(y, x, training_frame, ntrees, max_depth, min_rows,     
                             learn_rate, distribution)  
    * `y` : 종속변수   
    * `x` : 독립변수  
    * `training_frame` : 훈련데이터  
    * `ntrees` : 나무의 가지수. default는 50  
    * `max_depth` : 나무의 최대 높이. default는 5  
    * `min_rows` : leaf node에 주는 가중치의 최소값 default는 10  
    * `learn_rate` : learning rate. default는 0.1  
    * `distribution` : 분포 결정.  
    * 이 밖에도 여러가지 option이 있음.  

Model detail, training data의 MSE, Scoring History, 등을 알 수 있다.  
 
```
data(iris)
iris.hex <- as.h2o(iris, destination_frame = "iris.hex")

# 설명변수가 연속형일 때
iris.gbm <- h2o.gbm(y = 1, x = 2:5, training_frame = iris.hex, ntrees = 10,
                    max_depth = 3, min_rows = 2, learn_rate = 0.2, distribution = "gaussian")
summary(iris.gbm)   # model의 결과 & 정확도 출력

iris.gbm@model$scoring_history # tree의 수에 따른 rmse, mae, 분산출력
plot(iris.gbm)                 # tree의 수에 따른 error 그래프 출력

# 설명변수가 범주형일 때
iris.gbm2 <- h2o.gbm(y = 5, x = 1:4, training_frame = iris.hex, ntrees = 15, 
                     max_depth = 5, min_rows = 2, learn_rate = 0.01, 
                     distribution = "multinomial")
iris.gbm2@model$training_metrics
```

* 설명변수가 연속형일 때는 ntree에 따른 rsme 등 오차율을 확인할 수 있다.  
* 설명변수가 범주형일때는 training_metrics를 통해 confusion matrix, rsme, ratio등을 확인할 수 있다.  

<br>

### 4.2 Generalized Linear Models (GLM)

일반화 선형모델을 말하는 것으로 종속변수가 범주형이거나, 이산형일 경우 사용한다.  
보통 distribution으로 exponential 함수나 Gaussian 함수를 사용한다.   
H2O에서는 일반화하는 과정을 elastic net penalty를 사용한다.  
  
`hlo.gbm` function : h2o.glm(y, x, training_frame, family, nfolds, alpha)  
    * `y` : 종속변수  
    * `x` : 독립변수  
    * `training_frame` : 훈련데이터  
    * `family` : 종속변수의 분포를 정의  
    * `nfold` : K-fold 교차검정을 하는 횟수를 정의.  
    * `alpha` : L1(LASSO)와 L2(Ridge)사이의 값 지정, 1일 경우 L1을 선택, 0일 경우 L2를 선택한다. 그 중간은 Elastic 이라 부른다.  
    * 이 밖에도 여러가지 option이 있음.  
    
결과창으로는 MSE, AUC(과적합 판단), R^2(결정계수), Confusion Matrix 등을 보여준다.  

```
prostate.hex <- h2o.importFile(path = "https://raw.github.com/h2oai/h2o/master/smalldata/logreg/prostate.csv",
                               destination_frame = "prostate.hex")

prostate.glm <- h2o.glm(y = "CAPSULE", x = c("AGE", "RACE", "PSA", "DCAPS"), training_frame = prostate.hex,
                        family = "binomial", nfolds = 10, alpha = 0.5)
summary(prostate.glm)                                   # 성능 및 정확도에 대해 보여주고 있다.
prostate.glm@model$cross_validation_metrics

```

<br>

### 4.3 K-means

`h2o.kmeans` function : h2o.kmeans(training_frame, k, x)  
    * training_frame : 학습할 데이터  
    * k : k의 개수  
    * x : 입력될 데이터 변수  

Centroid Statistics 가 출력된다.   

```
h2o.kmeans(training_frame = iris.hex, k = 3, x = 1:4)
```

<br>

### 4.4 Principal Components Analysis(PCA)

h2o에서는 주성분분석도 지원해준다.  
`h2o.prcomp` function : h2o.prcomp(training_frame, transform, k)  
    * `training_frame` : 학습데이터 입력  
    * `transform` : training data를 어떻게 변환할건지 정의.  
    * `k` : PCA 주성분의 숫자 지정  
  
```
ausPath <- system.file("extdata", "australia.csv", package = "h2o")
australia.hex <- h2o.importFile(path = ausPath)
australia.pca <- h2o.prcomp(training_frame = australia.hex, transform = "STANDARDIZE", k = 3)
australia.pca
```

<br>

### 4.5 Predictions

우리가 예측을 할때 확인해야 할 것은 아래와 같다.  
    * predict : 모델을 돌린 후 나오는 예측값들.  
    * Confusion Matrix : 알고리즘 후 예측값과 실제값들에 대한 table  
    * AUC Curve : 민감도와 관련된 Curve로 값이 클 수록 적합하다고 할 수 있음  
    * PCA Score : 주성분분석을 했을 때 나오는 값으로 보통 0.85 까지 오면 주성분의 개수를 멈춤.  
    
```
prostate.fit <- h2o.predict(object = prostate.glm, newdata = prostate.hex)
prostate.fit
```
    
참고 : Machine Learning with R and H2O (http://h2o.ai/resources/)
