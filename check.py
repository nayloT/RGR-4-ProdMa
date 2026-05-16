import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.integrate import simpson
import os


os.makedirs("images", exist_ok=True)

# Численная реализация ряда Фурье
def fourier_coeffs_numerical(f, x, N, method='simpson'):
    """
    Вычисляет коэффициенты Фурье a0, an, bn численно.
    f: функция
    x: массив точек
    N: число гармоник
    """
    L = np.pi
    y = f(x)

    if method == 'simpson':
        a0 = (1 / L) * simpson(y, x)
    else:
        a0 = (1 / L) * np.trapz(y, x)

    a = np.zeros(N + 1)
    b = np.zeros(N + 1)

    for n in range(1, N + 1):
        cos_term = y * np.cos(n * x)
        sin_term = y * np.sin(n * x)
        if method == 'simpson':
            a[n] = (1 / L) * simpson(cos_term, x)
            b[n] = (1 / L) * simpson(sin_term, x)
        else:
            a[n] = (1 / L) * np.trapz(cos_term, x)
            b[n] = (1 / L) * np.trapz(sin_term, x)

    return a0, a, b


def partial_sum(x, a0, a, b, N):
    """Частичная сумма ряда Фурье"""
    s = a0 / 2 * np.ones_like(x)
    max_n = min(N, len(a) - 1, len(b) - 1)
    for n in range(1, max_n + 1):
        s += a[n] * np.cos(n * x) + b[n] * np.sin(n * x)
    return s

# Аналитические разложения функций
# Функция 1: f(x) = x
def f1(x):
    return x


def analytical_coeffs_f1(N):
    """Аналитические коэффициенты для f(x)=x"""
    a = np.zeros(N + 1)
    b = np.zeros(N + 1)
    for n in range(1, N + 1):
        b[n] = 2 * (-1) ** (n + 1) / n
    return 0, a, b


# Функция 2: f(x) = |x|
def f2(x):
    return np.abs(x)


def analytical_coeffs_f2(N):
    """Аналитические коэффициенты для f(x)=|x|"""
    a = np.zeros(N + 1)
    b = np.zeros(N + 1)
    a0 = np.pi
    for n in range(1, N + 1):
        if n % 2 == 0:
            a[n] = 0
        else:
            a[n] = -4 / (np.pi * n ** 2)
    return a0, a, b


# Функция 3: f(x) = sign(x)
def f3(x):
    return np.sign(x)


def analytical_coeffs_f3(N):
    """Аналитические коэффициенты для f(x)=sign(x)"""
    a = np.zeros(N + 1)
    b = np.zeros(N + 1)
    a0 = 0
    for n in range(1, N + 1):
        if n % 2 == 0:
            b[n] = 0
        else:
            b[n] = 4 / (n * np.pi)
    return a0, a, b

# 2.3 Быстрое преобразование Фурье
def compute_fft_coeffs(y, dt):
    n = len(y)
    yf = fft(y)
    freqs = fftfreq(n, dt)
    return freqs[:n // 2], 2.0 / n * np.abs(yf[:n // 2])

# Исследование сходимости для всех трех функций
x = np.linspace(-np.pi, np.pi, 1000)
functions = [
    (f1, analytical_coeffs_f1, "f(x)=x", "f1"),
    (f2, analytical_coeffs_f2, "f(x)=|x|", "f2"),
    (f3, analytical_coeffs_f3, "f(x)=sign(x)", "f3")
]

# Для каждой функции строим графики
for f, analytical_func, title, filename in functions:
    print(f"Обработка: {title}")

    y_true = f(x)

    # Численные коэффициенты
    a0_num, a_num, b_num = fourier_coeffs_numerical(f, x, N=100)
    # Аналитические коэффициенты
    a0_an, a_an, b_an = analytical_func(100)

    # Частичные суммы при разных N
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

    # Эффект Гиббса (только для функций с разрывами)
    if filename in ["f1", "f3"]:  # f1 имеет разрыв на границах, f3 - разрыв в 0
        plt.figure(figsize=(10, 6))
        N_gibbs = [10, 50, 200]
        for N in N_gibbs:
            y_an = partial_sum(x, a0_an, a_an, b_an, N)
            plt.plot(x, y_an, label=f'N={N}')
        plt.plot(x, y_true, 'k--', linewidth=2, label='Исходная')

        # Увеличиваем область вблизи разрыва
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
        err_num = np.sqrt(np.mean((y_true - y_num) ** 2))
        err_an = np.sqrt(np.mean((y_true - y_an) ** 2))
        errors_num.append(err_num)
        errors_an.append(err_an)

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
    a_num_plot = [a_num[n] for n in n_plot]
    a_an_plot = [a_an[n] for n in n_plot]
    b_num_plot = [b_num[n] for n in n_plot]
    b_an_plot = [b_an[n] for n in n_plot]

    axes[0].stem(n_plot, a_num_plot, linefmt='r-', markerfmt='ro', label='Численные')
    axes[0].stem(n_plot, a_an_plot, linefmt='b--', markerfmt='bs', label='Аналитические')
    axes[0].set_title('Коэффициенты a_n')
    axes[0].set_xlabel('n')
    axes[0].set_ylabel('a_n')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].stem(n_plot, b_num_plot, linefmt='r-', markerfmt='ro', label='Численные')
    axes[1].stem(n_plot, b_an_plot, linefmt='b--', markerfmt='bs', label='Аналитические')
    axes[1].set_title('Коэффициенты b_n')
    axes[1].set_xlabel('n')
    axes[1].set_ylabel('b_n')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(f'images/{filename}_coeffs_comparison.png', dpi=150)
    plt.close()

# Применение к реальному датасету (температура)
print("Обработка: реальный датасет")

# Генерируем реалистичный временной ряд
np.random.seed(42)
days = np.arange(0, 730)
T_base = 15
T_amplitude = 10
T = T_base + T_amplitude * np.cos(2 * np.pi * days / 365) + 2 * np.cos(4 * np.pi * days / 365) + np.random.normal(0, 2,
                                                                                                                  len(days))

plt.figure(figsize=(12, 4))
plt.plot(days, T)
plt.title('Исходный временной ряд (симулированные данные температуры)')
plt.xlabel('Дни')
plt.ylabel('Температура, °C')
plt.grid(True)
plt.savefig('images/dataset_raw.png', dpi=150)
plt.close()

# FFT и спектр
dt = 1.0
freqs, spectrum = compute_fft_coeffs(T, dt)

plt.figure(figsize=(10, 5))
plt.stem(freqs[:50], spectrum[:50], basefmt=" ")
plt.title('Спектр мощности (первые 50 частот)')
plt.xlabel('Частота (1/день)')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.savefig('images/fft_spectrum.png', dpi=150)
plt.close()

# Фильтрация — оставляем только 2 главные гармоники
T_fft = fft(T)
freqs_all = fftfreq(len(T), dt)
idx1 = np.argmin(np.abs(freqs_all - 1 / 365))
idx2 = np.argmin(np.abs(freqs_all - 2 / 365))
T_filtered_fft = np.zeros_like(T_fft, dtype=complex)
T_filtered_fft[idx1] = T_fft[idx1]
T_filtered_fft[-idx1] = T_fft[-idx1]
T_filtered_fft[idx2] = T_fft[idx2]
T_filtered_fft[-idx2] = T_fft[-idx2]
T_filtered = np.real(np.fft.ifft(T_filtered_fft))

plt.figure(figsize=(12, 4))
plt.plot(days, T, alpha=0.5, label='Исходный сигнал')
plt.plot(days, T_filtered, 'r', linewidth=2, label='Фильтрованный (год.+полугод.)')
plt.title('Фильтрация шума — выделение основных гармоник')
plt.xlabel('Дни')
plt.ylabel('Температура, °C')
plt.legend()
plt.grid(True)
plt.savefig('images/filtered_vs_original.png', dpi=150)
plt.close()

# Восстановление по ограниченному числу гармоник
N_harmonics_list = [2, 5, 20]
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
for ax, K in zip(axes, N_harmonics_list):
    T_partial_fft = np.zeros_like(T_fft, dtype=complex)
    amps = np.abs(T_fft[:len(T) // 2])
    top_k_idx = np.argsort(amps)[-K:]
    for idx in top_k_idx:
        T_partial_fft[idx] = T_fft[idx]
        T_partial_fft[-idx] = T_fft[-idx]
    T_recon = np.real(np.fft.ifft(T_partial_fft))
    ax.plot(days, T, alpha=0.5, label='Исходный')
    ax.plot(days, T_recon, 'r', label=f'Восстановлен, K={K}')
    ax.set_title(f'Восстановление сигнала по {K} гармоникам')
    ax.legend()
    ax.grid(True)
plt.tight_layout()
plt.savefig('images/reconstruction_comparison.png', dpi=150)
plt.close()

print("\nВсе графики сохранены в папку images/")
print("Сгенерированные файлы:")
for f in os.listdir("images"):
    print(f"  - {f}")
