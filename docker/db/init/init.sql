CREATE TABLE IF NOT EXISTS usuarios (
    id_usuario UUID PRIMARY KEY,
    nombre TEXT,
    sexo CHAR(1),
    edad INT,
    ocupacion TEXT,
    personalidad TEXT,
    p_actividad FLOAT
);

CREATE TABLE IF NOT EXISTS entradas_diario (
    id_entrada UUID PRIMARY KEY,
    id_usuario UUID REFERENCES usuarios(id_usuario),
    fecha DATE,
    texto_diario TEXT,
    emocion_principal TEXT,
    sentimiento_usuario TEXT
);
