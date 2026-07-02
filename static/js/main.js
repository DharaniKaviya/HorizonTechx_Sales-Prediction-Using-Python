/* =============================================
   main.js — Sales Prediction Dashboard
   ============================================= */

// ── Animated Counter ─────────────────────────
function animateCounter(el, target, duration = 1800, prefix = '', suffix = '') {
  const start = 0;
  const startTime = performance.now();

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
    const current = Math.floor(eased * target);
    el.textContent = prefix + current.toLocaleString('en-IN') + suffix;
    if (progress < 1) requestAnimationFrame(update);
  }

  requestAnimationFrame(update);
}

function initCounters() {
  document.querySelectorAll('[data-counter]').forEach(el => {
    const target   = parseFloat(el.dataset.counter) || 0;
    const prefix   = el.dataset.prefix  || '';
    const suffix   = el.dataset.suffix  || '';
    const duration = parseInt(el.dataset.duration) || 1800;

    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          animateCounter(el, target, duration, prefix, suffix);
          observer.disconnect();
        }
      });
    }, { threshold: 0.5 });

    observer.observe(el);
  });
}

// ── Scroll Reveal ─────────────────────────────
function initScrollReveal() {
  const reveals = document.querySelectorAll('.reveal');
  const observer = new IntersectionObserver(entries => {
    entries.forEach((entry, i) => {
      if (entry.isIntersecting) {
        setTimeout(() => {
          entry.target.classList.add('visible');
        }, i * 80);
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -60px 0px' });

  reveals.forEach(el => observer.observe(el));
}

// ── Navbar Mobile Toggle ──────────────────────
function initNavbar() {
  const toggle = document.getElementById('navToggle');
  const links  = document.getElementById('navLinks');

  if (toggle && links) {
    toggle.addEventListener('click', () => {
      links.classList.toggle('open');
    });

    // Close on outside click
    document.addEventListener('click', e => {
      if (!toggle.contains(e.target) && !links.contains(e.target)) {
        links.classList.remove('open');
      }
    });
  }

  // Active link highlighting
  const currentPath = window.location.pathname;
  document.querySelectorAll('.nav-links a').forEach(a => {
    if (a.getAttribute('href') === currentPath ||
        (currentPath === '/' && a.getAttribute('href') === '/')) {
      a.classList.add('active');
    }
  });
}

// ── Sticky Navbar shadow ──────────────────────
function initNavbarScroll() {
  const navbar = document.querySelector('.navbar');
  if (!navbar) return;

  window.addEventListener('scroll', () => {
    if (window.scrollY > 40) {
      navbar.style.boxShadow = '0 4px 30px rgba(0,0,0,0.5)';
    } else {
      navbar.style.boxShadow = 'none';
    }
  }, { passive: true });
}

// ── Image Lightbox ────────────────────────────
function initLightbox() {
  const overlay = document.getElementById('lightboxOverlay');
  const img     = document.getElementById('lightboxImg');
  const closeBtn= document.getElementById('lightboxClose');

  if (!overlay) return;

  document.querySelectorAll('.chart-item img').forEach(chartImg => {
    chartImg.style.cursor = 'zoom-in';
    chartImg.addEventListener('click', () => {
      img.src = chartImg.src;
      overlay.classList.add('active');
      document.body.style.overflow = 'hidden';
    });
  });

  function closeLightbox() {
    overlay.classList.remove('active');
    document.body.style.overflow = '';
  }

  if (closeBtn) closeBtn.addEventListener('click', closeLightbox);
  overlay.addEventListener('click', e => {
    if (e.target === overlay) closeLightbox();
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeLightbox();
  });
}

// ── Prediction Form ───────────────────────────
function initPredictForm() {
  const form   = document.getElementById('predictForm');
  const btn    = document.getElementById('predictBtn');
  if (!form || !btn) return;

  form.addEventListener('submit', function(e) {
    // Validate all required numeric inputs ≥ 0
    let valid = true;
    form.querySelectorAll('input[type="number"]').forEach(input => {
      const val = parseFloat(input.value);
      if (isNaN(val) || val < 0) {
        input.style.borderColor = 'var(--pink)';
        valid = false;
      } else {
        input.style.borderColor = '';
      }
    });

    if (!valid) {
      e.preventDefault();
      showToast('⚠️ Please enter valid positive values for all budget fields.', 'error');
      return;
    }

    // Loading state
    btn.innerHTML = '<span class="spinner"></span> Predicting...';
    btn.disabled = true;
  });

  // Real-time total spend calculation
  const spendInputs = ['tv_spend','radio_spend','newspaper_spend','social_spend','digital_spend'];
  const totalEl = document.getElementById('totalSpend');

  function updateTotal() {
    if (!totalEl) return;
    const total = spendInputs.reduce((sum, id) => {
      const el = document.getElementById(id);
      return sum + (el ? parseFloat(el.value) || 0 : 0);
    }, 0);
    totalEl.textContent = '₹ ' + total.toFixed(1) + ' K';
  }

  spendInputs.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', updateTotal);
  });

  updateTotal();
}

// ── Toast notification ────────────────────────
function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.style.cssText = `
    position: fixed; bottom: 24px; right: 24px;
    padding: 14px 22px;
    background: ${type === 'error' ? 'rgba(255,107,157,0.15)' : 'rgba(91,95,255,0.15)'};
    border: 1px solid ${type === 'error' ? 'rgba(255,107,157,0.5)' : 'rgba(91,95,255,0.5)'};
    border-radius: 10px;
    color: ${type === 'error' ? '#FF6B9D' : '#F0F0FF'};
    font-size: 0.9rem;
    font-weight: 500;
    backdrop-filter: blur(16px);
    z-index: 9999;
    animation: slideUp 0.3s ease;
    max-width: 360px;
  `;
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

// ── Hero Chart.js ─────────────────────────────
function initHeroChart() {
  const canvas = document.getElementById('heroChart');
  if (!canvas) return;

  const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const predicted = [420,480,510,560,590,620,610,650,710,820,960,1100];
  const actual    = [400,460,490,540,570,600,590,635,690,800,930,1060];

  new Chart(canvas, {
    type: 'line',
    data: {
      labels: months,
      datasets: [
        {
          label: 'Predicted Sales',
          data: predicted,
          borderColor: '#5B5FFF',
          backgroundColor: 'rgba(91,95,255,0.15)',
          borderWidth: 2.5,
          pointRadius: 4,
          pointBackgroundColor: '#5B5FFF',
          fill: true,
          tension: 0.4,
        },
        {
          label: 'Actual Sales',
          data: actual,
          borderColor: '#8A2BE2',
          backgroundColor: 'rgba(138,43,226,0.08)',
          borderWidth: 2,
          pointRadius: 3,
          pointBackgroundColor: '#8A2BE2',
          fill: true,
          tension: 0.4,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: {
          labels: { color: '#A0A0C0', font: { size: 11 }, boxWidth: 12 }
        },
        tooltip: {
          backgroundColor: 'rgba(15,15,30,0.95)',
          borderColor: 'rgba(91,95,255,0.4)',
          borderWidth: 1,
          titleColor: '#F0F0FF',
          bodyColor: '#A0A0C0',
          callbacks: {
            label: ctx => ` ₹ ${ctx.parsed.y.toLocaleString()} K`,
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(91,95,255,0.1)' },
          ticks: { color: '#666688', font: { size: 10 } }
        },
        y: {
          grid: { color: 'rgba(91,95,255,0.1)' },
          ticks: { color: '#666688', font: { size: 10 },
                   callback: v => '₹' + v + 'K' }
        }
      }
    }
  });
}

// ── Dashboard Charts ──────────────────────────
function initDashboardCharts() {
  // Platform bar chart
  const platformCanvas = document.getElementById('platformChart');
  if (platformCanvas) {
    const platformData = JSON.parse(platformCanvas.dataset.values || '{}');
    new Chart(platformCanvas, {
      type: 'bar',
      data: {
        labels: Object.keys(platformData),
        datasets: [{
          label: 'Avg Sales (₹K)',
          data: Object.values(platformData),
          backgroundColor: [
            'rgba(91,95,255,0.7)','rgba(138,43,226,0.7)',
            'rgba(0,212,255,0.7)','rgba(255,107,157,0.7)',
            'rgba(255,179,71,0.7)','rgba(0,230,118,0.7)'
          ],
          borderRadius: 6,
          borderWidth: 0,
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(15,15,30,0.95)',
            borderColor: 'rgba(91,95,255,0.4)',
            borderWidth: 1,
            callbacks: { label: ctx => ` ₹ ${ctx.parsed.y.toFixed(0)} K` }
          }
        },
        scales: {
          x: { grid: { display: false }, ticks: { color: '#666688' } },
          y: { grid: { color: 'rgba(91,95,255,0.1)' }, ticks: { color: '#666688', callback: v => '₹'+v+'K' } }
        }
      }
    });
  }

  // Monthly trend doughnut
  const monthCanvas = document.getElementById('monthChart');
  if (monthCanvas) {
    const monthData = JSON.parse(monthCanvas.dataset.values || '{}');
    new Chart(monthCanvas, {
      type: 'doughnut',
      data: {
        labels: Object.keys(monthData),
        datasets: [{
          data: Object.values(monthData),
          backgroundColor: [
            '#5B5FFF','#6E72FF','#8A2BE2','#A044FF','#FF6B9D',
            '#FFB347','#00D4FF','#00E676','#FF5252','#FFEB3B','#40C4FF','#7C4DFF'
          ],
          borderWidth: 0,
          hoverOffset: 8,
        }]
      },
      options: {
        responsive: true,
        cutout: '65%',
        plugins: {
          legend: { position: 'right', labels: { color: '#A0A0C0', font: { size: 10 }, boxWidth: 10, padding: 8 } },
          tooltip: {
            backgroundColor: 'rgba(15,15,30,0.95)',
            borderColor: 'rgba(91,95,255,0.4)',
            borderWidth: 1,
            callbacks: { label: ctx => ` ₹ ${ctx.parsed.toFixed(0)} K` }
          }
        }
      }
    });
  }
}

// ── R² bar animation ──────────────────────────
function initR2Bars() {
  document.querySelectorAll('.r2-fill').forEach(bar => {
    const width = bar.style.width;
    bar.style.width = '0%';
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          setTimeout(() => { bar.style.width = width; }, 200);
          observer.disconnect();
        }
      });
    });
    observer.observe(bar);
  });
}

// ── Result page confetti-style effect ─────────
function initResultEffect() {
  const result = document.querySelector('.result-value');
  if (!result) return;

  const particles = 20;
  for (let i = 0; i < particles; i++) {
    const p = document.createElement('div');
    p.style.cssText = `
      position: fixed;
      width: ${Math.random() * 8 + 4}px;
      height: ${Math.random() * 8 + 4}px;
      border-radius: 50%;
      background: ${['#5B5FFF','#8A2BE2','#00D4FF','#FF6B9D','#FFB347'][Math.floor(Math.random()*5)]};
      left: ${Math.random() * 100}vw;
      top: -20px;
      opacity: ${Math.random() * 0.8 + 0.2};
      animation: fall ${Math.random() * 2 + 2}s ease-in forwards;
      animation-delay: ${Math.random() * 1}s;
      pointer-events: none;
      z-index: 9999;
    `;
    document.body.appendChild(p);
    setTimeout(() => p.remove(), 4000);
  }

  const style = document.createElement('style');
  style.textContent = `
    @keyframes fall {
      to { transform: translateY(110vh) rotate(${Math.random()*720}deg); opacity: 0; }
    }
  `;
  document.head.appendChild(style);
}

// ── Init All ──────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initNavbar();
  initNavbarScroll();
  initScrollReveal();
  initCounters();
  initLightbox();
  initPredictForm();
  initHeroChart();
  initDashboardCharts();
  initR2Bars();
  initResultEffect();
});
