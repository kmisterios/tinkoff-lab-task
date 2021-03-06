# Тестовое задание для Tinkoff Lab

_Возьмите задачи SST-2, RTE и CoLA из GLUE и обучите на них классификатор. Попробуйте выжать максимум из качества._

**Отчет:** [Report.pdf](Report.pdf)

### Чтобы запустить код нужно

1\. Склонировать репозиторий

<code>git clone https://github.com/kmisterios/tinkoff-lab-task.git</code>

2\. Создать виртуальное окружение, активировать его, перейти в папку с кодом

<code>python -m venv myenv</code>,
<code>source myenv/bin/activate</code>,
<code>cd tinkoff-lab-task</code>

3\. Установить библиотеки

<code>pip install -r requirements.txt</code>

4\. Добавить ядро для Jupyter в окружении

<code>python -m ipykernel install --user --name=myenv</code>

5\. Создать папку в корне для сохранения весов модели

<code>mkdir checkpoints</code>

6\. Запустить Jupyter Notebook или аналог

<code>jupyter notebook</code> / <code>jupyter lab</code>

7\. Открыть ноутбук <code>Run_Experiments.ipynb</code>, убедиться, что выбрано правильное ядро, и запустить ячейки


