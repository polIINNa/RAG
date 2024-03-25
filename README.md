# Список Постановлений, по которым можно вести поиск:
26, 141, 158, 295, 566, 574, 785, 811, 895, 1302, 1528, 1570, 1598, 2186, 2221

# Сервис работы с RAG через FastAPI

## Запуск сервиса
```shell
docker compose -f fast-api-docker-compose.yaml up --build -d
```
- Логи в /var/log/rag_service/log/

## API сервиса
1. Проверка работоспособности сервиса:
```shell
curl http://'ip_service':8083/api/v1/healthcheck
```
2. Вопрос по документам по господдержке:
```shell
curl --header "Content-Type: application/json" --request POST --data '{"body":"question"}' http://'ip_service':8083/api/v1/question
```
Или из json-файла:
```shell
curl --header "Content-Type: application/json" --request POST --data-binary @"json_filename" http://'ip_service':8083/api/v1/question
```
