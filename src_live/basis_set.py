import numpy as np
from dataclasses import dataclass, field
import requests
import json
from .atom import ELEMENT_SYMBOL

ODD_DF = np.array([1, 1, 2, 3, 8, 15, 48, 105, 384, 945])
PI = np.pi

def l_to_ijk(L):
    IJK = []
    for I in range(L, -1 , -1):
        for J in range(L - I, -1, -1):
            IJK.append((I, J, L - I - J))
    return sorted(IJK, reverse = True)

def primitive_cart_norm(alpha, l, m, n):
    L = l + m + n
    num = (2 * alpha/PI)**(3/4) * (4 * alpha)**(L/2)
    den = np.sqrt(ODD_DF[l] * ODD_DF[m] * ODD_DF[n]) # hier wird die normierungsfaktor berechnet, der dafür sorgt, dass die gaussfunktion normiert ist, also dass das integral über alle raum gleich 1 ist
    return num/den

@dataclass
class Shell:
    l : int # 0 s, 1 p, 2 d, 3 f 
    exponents: np.ndarray # 1d array of exponents
    coefficients: np.ndarray # 1d array of coefficients
    norm_factors : np.ndarray = field(init=False) # wird nicht im init constructor übergeben, sondern wird automatisch berechnet
    #center: np.ndarray = field(init=False)

    def __post_init__(self):
        self.norm_factors = self.get_norm_factors()

    def get_norm_factors(self):
        ijk = l_to_ijk(self.l)
        norm_factors = np.empty((len(self.exponents), len(ijk)), dtype = np.float64)
        for i, (l, m, n) in enumerate(ijk):
            norm_factors[:,i] = primitive_cart_norm(self.exponents, l, m, n)
        return norm_factors
    
    def set_center(self, center: np.ndarray) -> None:
        object.__setattr__(self, "center", np.asarray(center, dtype=float))

@dataclass  # basissatz aus dictonary aus shells
class BasisSet:
    name: str
    elements: dict[str, list[Shell]] = field(default_factory=dict) # default_factory ist eine Funktion, die aufgerufen wird, um den Standardwert für das Feld zu erstellen. In diesem Fall wird eine leere dict erstellt, wenn kein Wert übergeben wird.
    
    def download_from_bse(self, element_list: list, timeout: int = 30) -> None:
        url = f"https://www.basissetexchange.org/api/basis/{self.name}/format/json/"
        url += f"?elements={','.join(element_list)}"
        response = requests.get(url, timeout=timeout)
        data_json = response.json()
        print(f"Dump data to {self.name}.json")
        json.dump(data_json, open(f"{self.name}.json", "w"), indent = 4)
        self.parse_elements(data_json)

    def parse_elements(self, data: dict) -> None:
        self.elements = {} # leeres dict, das mit den neuen Daten gefüllt wird
        basis_data = data.get("elements", {}) # get methode von dict, die den wert für den key "elements" zurückgibt, oder ein leeres dict, wenn der key nicht existiert
        for atomic_number, element_data in basis_data.items():
            symbol = ELEMENT_SYMBOL.get(int(atomic_number))
            shells: list[Shell] = []
            for shell_data in element_data.get("electron_shells", []):
                angular_momentum = shell_data["angular_momentum"]
                contraction_coefficients = shell_data["coefficients"]
                exponents = shell_data["exponents"]
                for l, coefficients in zip(angular_momentum, contraction_coefficients):
                    shells.append(Shell(l = int(l),
                                        exponents = np.asarray(exponents, dtype=float),
                                        coefficients = np.asarray(coefficients, dtype=float)))
                self.elements[symbol] = shells

    def __str__(self) -> str:
        output = f"Basis set: {self.name}\n"
        for symbol, shells in self.elements.items():
            output += f"Element: {symbol}\n"
            for shell in shells:
                output += f"l = {shell.l}, exponents = {shell.exponents}, coefficients = {shell.coefficients}\n"
                output += f"norm_factors = {shell.norm_factors}"
        return output
                
