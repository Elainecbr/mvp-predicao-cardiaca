FROM nginx:1.27-alpine

COPY index.html styles.css scripts.js /usr/share/nginx/html/
RUN mkdir -p /usr/share/nginx/html/img
COPY --chmod=644 img/consulta-cardiovascular.png /usr/share/nginx/html/img/consulta-cardiovascular.png

EXPOSE 80
