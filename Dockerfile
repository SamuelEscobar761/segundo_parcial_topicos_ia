# Usa una imagen base de Python
FROM python:3.9

# Establece el directorio de trabajo en el contenedor
WORKDIR /segundo_parcial_topicos_ia

# Copia los archivos necesarios al contenedor
COPY ./requirements.txt /segundo_parcial_topicos_ia/requirements.txt

# Instala las dependencias
RUN pip install --no-cache-dir --upgrade -r /segundo_parcial_topicos_ia/requirements.txt

COPY ./app /segundo_parcial_topicos_ia/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
