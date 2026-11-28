const form = document.querySelector("#formulario");
const result = document.querySelector("#resultado");
const submitButton = form.querySelector('button[type="submit"]');
const numericFields = new Set(["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]);

document.querySelector("#preencher-exemplo").addEventListener("click", () => {
  const example = {
    Age: "54", Sex: "F", ChestPainType: "NAP", RestingBP: "130",
    Cholesterol: "240", FastingBS: "0", RestingECG: "Normal",
    MaxHR: "150", ExerciseAngina: "N", Oldpeak: "1.5", ST_Slope: "Up",
  };
  for (const [name, value] of Object.entries(example)) {
    const field = form.elements.namedItem(name);
    field.value = value;
  }
  result.hidden = true;
  result.textContent = "";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!form.reportValidity()) return;

  const data = Object.fromEntries(new FormData(form));
  for (const field of numericFields) data[field] = Number(data[field]);
  result.hidden = false;
  result.className = "result";
  result.textContent = "Processando os dados...";
  submitButton.disabled = true;
  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || "A API não conseguiu processar os dados.");
    if (body.prediction !== 0 && body.prediction !== 1) throw new Error("A API retornou um resultado inesperado.");
    const classification = body.prediction === 1 ? "classe 1 (doença cardíaca)" : "classe 0 (sem doença cardíaca)";
    result.textContent = `Resultado experimental: ${classification}. Esta classificação não é um diagnóstico, nem uma estimativa de risco futuro. Não utilize para decisões de saúde; procure um profissional.`;
  } catch (error) {
    result.classList.add("error");
    result.textContent = `Não foi possível concluir a avaliação: ${error.message}. Confira se a API está em execução.`;
  } finally {
    submitButton.disabled = false;
  }
});
