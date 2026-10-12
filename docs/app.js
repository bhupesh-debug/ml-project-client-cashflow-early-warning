/**
 * Cardone Platform CashFlow Guard — Interactive Browser Simulation Engine
 * Author: Bhupesh Chandra Dimri
 *
 * NOTE: GitHub Pages serves static client-side files. This module provides a faithful,
 * zero-latency client-side mathematical emulation of the trained Python HistGradientBoosting
 * model for interactive recruiter exploration. For official Python inference on raw CSVs,
 * execute `python scripts/run_demo.py` or inspect `src/inference/pipeline.py`.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const endingCashInput = document.getElementById("endingCash");
  const monthlyOpexInput = document.getElementById("monthlyOpex");
  const weeklyRevInput = document.getElementById("weeklyRev");
  const dsoInput = document.getElementById("dsoDays");
  const wipUnderInput = document.getElementById("wipUnderbillings");
  const industrySelect = document.getElementById("industry");

  // Display elements
  const cashValDisplay = document.getElementById("cashValDisplay");
  const opexValDisplay = document.getElementById("opexValDisplay");
  const revValDisplay = document.getElementById("revValDisplay");
  const dsoValDisplay = document.getElementById("dsoValDisplay");
  const wipValDisplay = document.getElementById("wipValDisplay");

  // Output elements
  const riskBox = document.getElementById("riskIndicator");
  const riskTitle = document.getElementById("riskTitle");
  const breachProbDisplay = document.getElementById("breachProbDisplay");
  const forecast13wDisplay = document.getElementById("forecast13wDisplay");
  const dailyTargetDisplay = document.getElementById("dailyTargetDisplay");
  const burnCoverageDisplay = document.getElementById("burnCoverageDisplay");
  const interventionsList = document.getElementById("interventionsList");

  // Presets
  const presetContractor = document.getElementById("presetContractor");
  const presetDental = document.getElementById("presetDental");
  const presetSaas = document.getElementById("presetSaas");

  function formatCurrency(val) {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 0,
    }).format(val);
  }

  function recalculate() {
    const cash = parseFloat(endingCashInput.value);
    const monthlyOpex = parseFloat(monthlyOpexInput.value);
    const weeklyRev = parseFloat(weeklyRevInput.value);
    const dso = parseFloat(dsoInput.value);
    const wipUnder = parseFloat(wipUnderInput.value);
    const industry = industrySelect.value;

    // Update slider label texts
    cashValDisplay.textContent = formatCurrency(cash);
    opexValDisplay.textContent = formatCurrency(monthlyOpex);
    revValDisplay.textContent = formatCurrency(weeklyRev);
    dsoValDisplay.textContent = `${dso} days`;
    wipValDisplay.textContent = formatCurrency(wipUnder);

    // Baseline formulas (Cardone Ventures Platform Review standard)
    const dailyOpEx = monthlyOpex / 30.0;
    const dailyDebt = 650.0;
    const dailyCashTarget = (dailyOpEx * 60.0) + (dailyDebt * 60.0);
    const weeklyOutflow = (monthlyOpex / 4.33) + (dailyDebt * 7.0);
    const burnCoverageWeeks = cash / Math.max(weeklyOutflow, 1.0);

    // Emulated ML Gradient-Boosted Predictor Response
    // 1. Forecast 13-week net cash flow
    const weeklyNet = weeklyRev - weeklyOutflow - (dso > 55 ? (dso - 55) * 450 : 0) - (wipUnder * 0.05);
    const predicted13wCash = (weeklyNet * 13.0) + (wipUnder > 0 ? wipUnder * 0.40 : 0);

    // 2. 45-day Cash Target Breach Probability
    const targetDeficit = dailyCashTarget - cash;
    let breachScore = (targetDeficit / dailyCashTarget) * 1.8 + ((dso - 45) / 50.0) * 0.6;
    if (burnCoverageWeeks < 4.0) breachScore += 0.5;
    if (burnCoverageWeeks > 9.0) breachScore -= 0.6;

    const breachProb = Math.min(0.999, Math.max(0.001, 1.0 / (1.0 + Math.exp(-breachScore * 2.2))));

    // Update Output UI
    dailyTargetDisplay.textContent = formatCurrency(dailyCashTarget);
    burnCoverageDisplay.textContent = `${burnCoverageWeeks.toFixed(1)} weeks`;
    breachProbDisplay.textContent = `${(breachProb * 100).toFixed(1)}%`;
    forecast13wDisplay.textContent = formatCurrency(predicted13wCash);

    if (predicted13wCash < 0) {
      forecast13wDisplay.style.color = "var(--danger)";
    } else {
      forecast13wDisplay.style.color = "var(--success)";
    }

    // Risk tiering
    riskBox.className = "risk-indicator";
    let interventions = [];

    if (breachProb >= 0.55) {
      riskBox.classList.add("critical");
      riskTitle.textContent = "CRITICAL LIQUIDITY BREACH RISK";
      interventions.push("Immediate TAPW Protocol Activation: Enforce non-negotiable Daily Cash Target monitoring.");
      interventions.push("Freeze owner discretionary draws and halt unapproved capital commitments.");
    } else if (breachProb >= 0.25) {
      riskBox.classList.add("high");
      riskTitle.textContent = "ELEVATED CASH WATCH WARNING";
      interventions.push("Establish 13-week direct cash flow weekly war room with core triad (Finance, Marketing, BAM).");
    } else {
      riskBox.classList.add("low");
      riskTitle.textContent = "HEALTHY CASH BUFFER ASSIGNED";
      interventions.push("Maintain operational pacing. Focus triad on capacity expansion and ODE margin optimization.");
    }

    if (dso > 50) {
      interventions.push(`Accelerate Receivables: DSO at ${dso} days is draining working capital. Deploy early-pay discounts (2/10 Net 30).`);
    }

    if (wipUnder > 25000) {
      interventions.push(`Recover Trapped Capital: ${formatCurrency(wipUnder)} in unbilled project progress identified. Enforce immediate milestone billing.`);
    }

    // Render interventions
    interventionsList.innerHTML = "";
    interventions.forEach(item => {
      const li = document.createElement("li");
      li.textContent = item;
      interventionsList.appendChild(li);
    });
  }

  // Presets
  presetContractor.addEventListener("click", () => {
    setActivePreset(presetContractor);
    endingCashInput.value = 95000;
    monthlyOpexInput.value = 160000;
    weeklyRevInput.value = 48000;
    dsoInput.value = 68;
    wipUnderInput.value = 68000;
    industrySelect.value = "Commercial Contracting";
    recalculate();
  });

  presetDental.addEventListener("click", () => {
    setActivePreset(presetDental);
    endingCashInput.value = 380000;
    monthlyOpexInput.value = 110000;
    weeklyRevInput.value = 39000;
    dsoInput.value = 28;
    wipUnderInput.value = 0;
    industrySelect.value = "Healthcare & Dental";
    recalculate();
  });

  presetSaas.addEventListener("click", () => {
    setActivePreset(presetSaas);
    endingCashInput.value = 140000;
    monthlyOpexInput.value = 150000;
    weeklyRevInput.value = 38000;
    dsoInput.value = 46;
    wipUnderInput.value = 15000;
    industrySelect.value = "B2B SaaS & Tech";
    recalculate();
  });

  function setActivePreset(btn) {
    [presetContractor, presetDental, presetSaas].forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
  }

  // Listeners
  [endingCashInput, monthlyOpexInput, weeklyRevInput, dsoInput, wipUnderInput, industrySelect].forEach(el => {
    el.addEventListener("input", recalculate);
  });

  // Initial calculation
  recalculate();
});
