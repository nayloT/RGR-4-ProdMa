import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.integrate import simpson
import os


os.makedirs("images", exist_ok=True)


# Численная реализация ряда Фурье
def fourier_coeffs_numerical(f, x, N):
    L = np.pi
    y = f(x)

    a0 = (1 / L) * simpson(y, x)

    a = np.zeros(N + 1)
    b = np.zeros(N + 1)

    for n in range(1, N + 1):
        cos_term = y * np.cos(n * x)
        sin_term = y * np.sin(n * x)
        a[n] = (1 / L) * simpson(cos_term, x)
        b[n] = (1 / L) * simpson(sin_term, x)

    return a0, a, b


def partial_sum(x, a0, a, b, N):
    """Частичная сумма ряда Фурье"""
    s = a0 / 2 * np.ones_like(x)
    max_n = min(N, len(a) - 1, len(b) - 1)
    for n in range(1, max_n + 1):
        s += a[n] * np.cos(n * x) + b[n] * np.sin(n * x)
    return s


# Аналитические разложения функций

def f1(x):
    return x


def analytical_coeffs_f1(N):
    a = np.zeros(N + 1)
    b = np.zeros(N + 1)
    for n in range(1, N + 1):
        b[n] = 2 * (-1) ** (n + 1) / n
    return 0, a, b


def f2(x):
    return np.abs(x)


def analytical_coeffs_f2(N):
    a = np.zeros(N + 1)
    b = np.zeros(N + 1)
    a0 = np.pi
    for n in range(1, N + 1):
        if n % 2 == 0:
            a[n] = 0
        else:
            a[n] = -4 / (np.pi * n ** 2)
    return a0, a, b


def f3(x):
    return np.sign(x)


def analytical_coeffs_f3(N):
    a = np.zeros(N + 1)
    b = np.zeros(N + 1)
    a0 = 0
    for n in range(1, N + 1):
        if n % 2 == 0:
            b[n] = 0
        else:
            b[n] = 4 / (n * np.pi)
    return a0, a, b


# Исследование сходимости для всех трех функций

x = np.linspace(-np.pi, np.pi, 1000)
functions = [
    (f1, analytical_coeffs_f1, "f(x)=x", "f1"),
    (f2, analytical_coeffs_f2, "f(x)=|x|", "f2"),
    (f3, analytical_coeffs_f3, "f(x)=sign(x)", "f3")
]

for f, analytical_func, title, filename in functions:
    print(f"Обработка: {title}")
    y_true = f(x)

    a0_num, a_num, b_num = fourier_coeffs_numerical(f, x, N=100)
    a0_an, a_an, b_an = analytical_func(100)

    # Частичные суммы
    Ns = [5, 20, 100]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, N in enumerate(Ns):
        y_num = partial_sum(x, a0_num, a_num, b_num, N)
        y_an = partial_sum(x, a0_an, a_an, b_an, N)
        axes[idx].plot(x, y_true, 'k--', linewidth=2, label='Исходная')
        axes[idx].plot(x, y_num, 'r-', label=f'Численная, N={N}')
        axes[idx].plot(x, y_an, 'b:', label=f'Аналитическая, N={N}')
        axes[idx].set_title(f'{title}, N={N}')
        axes[idx].set_xlabel('x')
        axes[idx].set_ylabel('f(x)')
        axes[idx].legend()
        axes[idx].grid(True)
    plt.tight_layout()
    plt.savefig(f'images/{filename}_partial_sums.png', dpi=150)
    plt.close()

    # Эффект Гиббса
    if filename in ["f1", "f3"]:
        plt.figure(figsize=(10, 6))
        N_gibbs = [10, 50, 200]
        for N in N_gibbs:
            y_an = partial_sum(x, a0_an, a_an, b_an, N)
            plt.plot(x, y_an, label=f'N={N}')
        plt.plot(x, y_true, 'k--', linewidth=2, label='Исходная')
        if filename == "f1":
            plt.xlim(np.pi - 0.5, np.pi + 0.5)
            plt.title(f'Эффект Гиббса для {title} (вблизи x=π)')
        else:
            plt.xlim(-0.5, 0.5)
            plt.title(f'Эффект Гиббса для {title} (вблизи x=0)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'images/{filename}_gibbs_effect.png', dpi=150)
        plt.close()

    # Ошибка аппроксимации
    errors_num = []
    errors_an = []
    N_range = range(1, 101)
    for N in N_range:
        y_num = partial_sum(x, a0_num, a_num, b_num, N)
        y_an = partial_sum(x, a0_an, a_an, b_an, N)
        errors_num.append(np.sqrt(np.mean((y_true - y_num) ** 2)))
        errors_an.append(np.sqrt(np.mean((y_true - y_an) ** 2)))

    plt.figure(figsize=(8, 5))
    plt.semilogy(N_range, errors_num, 'r-', label='Численный метод')
    plt.semilogy(N_range, errors_an, 'b--', label='Аналитический метод')
    plt.title(f'Сходимость ряда Фурье: {title}')
    plt.xlabel('Число гармоник N')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'images/{filename}_error_analysis.png', dpi=150)
    plt.close()

    # Сравнение коэффициентов
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    n_plot = range(1, 21)
    axes[0].stem(n_plot, [a_num[n] for n in n_plot], linefmt='r-', markerfmt='ro', label='Численные')
    axes[0].stem(n_plot, [a_an[n] for n in n_plot], linefmt='b--', markerfmt='bs', label='Аналитические')
    axes[0].set_title(f'Коэффициенты a_n для {title}')
    axes[0].set_xlabel('n')
    axes[0].set_ylabel('a_n')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].stem(n_plot, [b_num[n] for n in n_plot], linefmt='r-', markerfmt='ro', label='Численные')
    axes[1].stem(n_plot, [b_an[n] for n in n_plot], linefmt='b--', markerfmt='bs', label='Аналитические')
    axes[1].set_title(f'Коэффициенты b_n для {title}')
    axes[1].set_xlabel('n')
    axes[1].set_ylabel('b_n')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(f'images/{filename}_coeffs_comparison.png', dpi=150)
    plt.close()

monthly_avg_spb = [-5.5, -5.0, -0.5, 5.5, 12.0, 16.5, 19.0, 17.5, 12.0, 6.0, 0.5, -3.0]

# Интерполируем на каждый день года (365 дней)
days = np.arange(0, 365)
month_centers = np.linspace(15, 345, 12)  # середины месяцев
temperature = np.interp(days, month_centers, monthly_avg_spb)

# Строим график
plt.figure(figsize=(12, 5))
plt.plot(days, temperature, 'g-', linewidth=1.5)
plt.title('Реальный временной ряд температуры (Санкт-Петербург, среднемесячные данные, без шума)')
plt.xlabel('Дни от начала года')
plt.ylabel('Температура, °C')
plt.grid(True)
plt.savefig('images/dataset_raw.png', dpi=150)
plt.close()

# Спектр мощности
dt = 1.0
n = len(temperature)
T_fft = fft(temperature)
freqs = fftfreq(n, dt)
positive_freqs = freqs[:n // 2]
amplitude_spectrum = 2.0 / n * np.abs(T_fft[:n // 2])

plt.figure(figsize=(10, 5))
plt.stem(positive_freqs[:30], amplitude_spectrum[:30], basefmt=" ")
plt.title('Спектр мощности температуры (Санкт-Петербург) — виден годовой цикл')
plt.xlabel('Частота (1/день)')
plt.ylabel('Амплитуда, °C')
plt.grid(True)
plt.savefig('images/fft_spectrum.png', dpi=150)
plt.close()

# Доминирующая частота
dominant_idx = np.argmax(amplitude_spectrum[1:]) + 1
dominant_freq = positive_freqs[dominant_idx]
dominant_period = 1 / dominant_freq
print(f"\nДоминирующая частота: {dominant_freq:.4f} 1/день")
print(f"Период: {dominant_period:.1f} дней (годовой цикл)")

# Фильтрация и восстановление
# Фильтрация — оставляем только 2 главные гармоники
idx1 = np.argmin(np.abs(positive_freqs - dominant_freq))
idx2 = np.argmin(np.abs(positive_freqs - 2 * dominant_freq))

T_filtered_fft = np.zeros_like(T_fft, dtype=complex)
T_filtered_fft[idx1] = T_fft[idx1]
T_filtered_fft[-idx1] = T_fft[-idx1]
if idx2 < len(positive_freqs):
    T_filtered_fft[idx2] = T_fft[idx2]
    T_filtered_fft[-idx2] = T_fft[-idx2]
T_filtered = np.real(np.fft.ifft(T_filtered_fft))

plt.figure(figsize=(12, 5))
plt.plot(days, temperature, alpha=0.7, label='Исходный сигнал (СПб)')
plt.plot(days, T_filtered, 'r', linewidth=2, label='Фильтрованный (годовая + полугодовая)')
plt.title('Фильтрация — выделение основных гармоник (Санкт-Петербург)')
plt.xlabel('Дни')
plt.ylabel('Температура, °C')
plt.legend()
plt.grid(True)
plt.savefig('images/filtered_vs_original.png', dpi=150)
plt.close()

# Восстановление по ограниченному числу гармоник
N_harmonics_list = [2, 5, 20]
fig, axes = plt.subplots(3, 1, figsize=(10, 9))
for ax, K in zip(axes, N_harmonics_list):
    T_partial_fft = np.zeros_like(T_fft, dtype=complex)
    amps = np.abs(T_fft[:n // 2])
    top_k_idx = np.argsort(amps)[-K:]
    for idx in top_k_idx:
        T_partial_fft[idx] = T_fft[idx]
        T_partial_fft[-idx] = T_fft[-idx]
    T_recon = np.real(np.fft.ifft(T_partial_fft))
    ax.plot(days, temperature, alpha=0.5, label='Исходный')
    ax.plot(days, T_recon, 'r', label=f'Восстановлен, K={K}')
    ax.set_title(f'Восстановление сигнала по {K} гармоникам (СПб)')
    ax.legend()
    ax.grid(True)
plt.tight_layout()
plt.savefig('images/reconstruction_comparison.png', dpi=150)
plt.close()

print("\nВсе графики сохранены в папку images/")
print("\nСгенерированные файлы:")
for f in sorted(os.listdir("images")):
    print(f"  - {f}")
