using System.Text.RegularExpressions;

namespace OContabil.Services;

public static class Validators
{
    /// <summary>
    /// Validates a Brazilian CNPJ using the mathematical algorithm (2 check digits).
    /// </summary>
    public static bool ValidateCnpj(string cnpj)
    {
        // Remove formatting
        cnpj = Regex.Replace(cnpj, @"[^\d]", "");

        if (cnpj.Length != 14)
            return false;

        // Reject all-same-digit CNPJs (e.g., 11111111111111)
        if (cnpj.Distinct().Count() == 1)
            return false;

        // Calculate first check digit
        int[] weights1 = { 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2 };
        int sum = 0;
        for (int i = 0; i < 12; i++)
            sum += (cnpj[i] - '0') * weights1[i];

        int remainder = sum % 11;
        int digit1 = remainder < 2 ? 0 : 11 - remainder;

        if ((cnpj[12] - '0') != digit1)
            return false;

        // Calculate second check digit
        int[] weights2 = { 6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2 };
        sum = 0;
        for (int i = 0; i < 13; i++)
            sum += (cnpj[i] - '0') * weights2[i];

        remainder = sum % 11;
        int digit2 = remainder < 2 ? 0 : 11 - remainder;

        return (cnpj[13] - '0') == digit2;
    }

    /// <summary>
    /// Format CNPJ as XX.XXX.XXX/XXXX-XX
    /// </summary>
    public static string FormatCnpj(string cnpj)
    {
        var digits = Regex.Replace(cnpj, @"[^\d]", "");
        if (digits.Length != 14) return cnpj;
        return $"{digits[..2]}.{digits[2..5]}.{digits[5..8]}/{digits[8..12]}-{digits[12..14]}";
    }

    /// <summary>
    /// Validates an email address format.
    /// </summary>
    public static bool ValidateEmail(string email)
    {
        if (string.IsNullOrWhiteSpace(email))
            return true; // Email is optional

        return Regex.IsMatch(email.Trim(),
            @"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$");
    }

    /// <summary>
    /// Validates a Brazilian phone number (10 or 11 digits).
    /// </summary>
    public static bool ValidatePhone(string phone)
    {
        if (string.IsNullOrWhiteSpace(phone))
            return true; // Phone is optional

        var digits = Regex.Replace(phone, @"[^\d]", "");

        // Brazilian phones: 10 digits (landline) or 11 digits (mobile)
        return digits.Length == 10 || digits.Length == 11;
    }

    /// <summary>
    /// Format phone as (XX) XXXXX-XXXX or (XX) XXXX-XXXX
    /// </summary>
    public static string FormatPhone(string phone)
    {
        var digits = Regex.Replace(phone, @"[^\d]", "");
        if (digits.Length == 11)
            return $"({digits[..2]}) {digits[2..7]}-{digits[7..11]}";
        if (digits.Length == 10)
            return $"({digits[..2]}) {digits[2..6]}-{digits[6..10]}";
        return phone;
    }
}
