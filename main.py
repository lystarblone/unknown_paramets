import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def model_func(k, a, omega, phi, b, c):
    return a * np.sin(omega * k + phi) + b * k + c

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

k_train = train['k'].values
x_train = train['x'].values

b_trend, c_trend = np.polyfit(k_train, x_train, 1)
trend = b_trend * k_train + c_trend
residuals = x_train - trend

n = len(k_train)
freqs = np.fft.fftfreq(n, d=1)
fft_vals = np.fft.fft(residuals)
fft_magnitude = np.abs(fft_vals)

dominant_freq_idx = np.argmax(fft_magnitude[1:n//2]) + 1
dominant_freq = abs(freqs[dominant_freq_idx])
omega_guess = 2 * np.pi * dominant_freq
print(f"Частота из FFT: {omega_guess:.6f}")

best_params = None
best_mse = float('inf')

amp_guess = (np.max(residuals) - np.min(residuals)) / 2

bounds = (
    [0, omega_guess*0.9, -2*np.pi, -np.inf, -np.inf],
    [amp_guess*5, omega_guess*1.1,  2*np.pi,  np.inf,  np.inf]
)

for a0 in [amp_guess, amp_guess*1.5, amp_guess*0.5]:
    for phi0 in np.linspace(0, 2*np.pi, 8):
        p0 = [a0, omega_guess, phi0, b_trend, c_trend]
        try:
            params, _ = curve_fit(
                model_func,
                k_train,
                x_train,
                p0=p0,
                bounds=bounds,
                maxfev=20000
            )
            mse = np.mean((model_func(k_train, *params) - x_train)**2)
            if mse < best_mse:
                best_mse = mse
                best_params = params
        except RuntimeError:
            continue

a, omega, phi, b, c = best_params
print(f"\nЛучшие параметры:\n"
      f"a = {a:.6f}, ω = {omega:.6f}, φ = {phi:.6f}, b = {b:.6f}, c = {c:.6f}")
print(f"Лучший MSE: {best_mse:.6f}")

k_test = test['k'].values
x_pred = model_func(k_test, *best_params)

pd.DataFrame({'k': k_test, 'x': x_pred}).to_csv('pred.csv', index=False)
print("pred.csv сохранён")

plt.figure(figsize=(10,5))
plt.scatter(k_train, x_train, s=10, label='Train data', color='blue')
plt.plot(k_train, model_func(k_train, *best_params), color='red', label='Model fit')
plt.xlabel('k')
plt.ylabel('x')
plt.legend()
plt.grid(True)
plt.savefig('fit_plot.png')
plt.close()