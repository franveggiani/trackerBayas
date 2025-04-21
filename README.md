# trackerBayas
============

Descripción
-----------
trackerBayas es una aplicación diseñada para rastrear y gestionar información relacionada con bayas. El proyecto está desarrollado principalmente en Python y utiliza Docker para facilitar su despliegue y ejecución.

Estructura del Proyecto
-----------------------
- api/: Contiene los archivos relacionados con la API del proyecto.
- requirements.txt: Lista de dependencias necesarias para ejecutar el proyecto.
- Dockerfile: Archivo para construir la imagen Docker de la aplicación.
- docker-compose.yml: Archivo para orquestar contenedores Docker relacionados con el proyecto.
- .gitignore: Especifica los archivos y directorios que Git debe ignorar.

Requisitos
----------
- Python 3.x
- Docker

Uso con Docker
--------------
1. Construir la imagen Docker:
```
    docker-compose build
```

2. Ejecutar el contenedor:
```
   docker-compose up
```
Ejemplo de petición
--------------
```
{
  "input_path": "./input",
  "id_racimo": "001",
  "video_name": "VID_20230322_173233",
  "output_path": "./output",
  "draw_tracking": true,
  "draw_circles": true,
  "radius": 10
}
```

Contribuciones
--------------
Las contribuciones son bienvenidas. Por favor, abre un issue o un pull request para discutir cambios importantes antes de implementarlos.

Licencia
--------
Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.

Contacto
--------
Desarrollado por franveggiani. Para más información, visita el repositorio en GitHub: https://github.com/franveggiani/trackerBayas
