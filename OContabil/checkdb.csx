using System;
using System.IO;
using System.Security.Cryptography;
using Microsoft.Data.Sqlite;

var dbPath = Path.Combine(
    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
    "OContabil", "ocontabil.db");

Console.WriteLine("DB: " + dbPath);

using var conn = new SqliteConnection("Data Source=" + dbPath);
conn.Open();

// Read current user
var cmd = conn.CreateCommand();
cmd.CommandText = "SELECT id, username, password_hash FROM users;";
using var reader = cmd.ExecuteReader();
while (reader.Read()) {
    Console.WriteLine($"User: {reader.GetString(1)}, Hash prefix: {reader.GetString(2)[..20]}");
}
